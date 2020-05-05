#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:57:49 2019

@author: johanm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical, Normal


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.args = args
        if args.model_type == 'ibp' and args.disc != 0:
            raise ValueError('In IBP model, no discrete variable is used, set args.disc to zero!')
        # encoder sub-model
        encode_layers = [nn.Conv2d(args.img_channel, 32, (4, 4), stride=2, padding=1), 
                         nn.BatchNorm2d(32),
                         nn.ReLU(True)]
        if args.img_size == 64:
            encode_layers += [nn.Conv2d(32, 32, (4, 4), stride=2, padding=1), 
                              nn.BatchNorm2d(32), 
                              nn.ReLU(True)]
        elif args.img_size == 32:
            pass
        else:
            raise RuntimeError('Not implemented for img_size={}'.format(args.img_size))
        encode_layers += [nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(True),
                          nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(True)]
        
        self.img2features = nn.Sequential(*encode_layers)
        self.features2hidden = nn.Sequential(nn.Linear(64*4*4, args.hidden_dim), nn.ReLU(True))
        
        self.hidden2pi = nn.Sequential(nn.Linear(args.hidden_dim, args.cont), nn.Sigmoid())
        self.hidden2phi = nn.Linear(args.hidden_dim, args.cont)
        self.hidden2psi = nn.Linear(args.hidden_dim, args.cont)
        self.hidden2taua = nn.Sequential(nn.Linear(args.hidden_dim, args.cont), nn.ReLU(True))
        self.hidden2taub = nn.Sequential(nn.Linear(args.hidden_dim, args.cont), nn.ReLU(True))
        
        if args.model_type == 'c-ibp':
            self.hidden2eta = nn.Sequential(nn.Linear(args.hidden_dim, args.disc), nn.Softmax(dim=-1))
        if args.model_type == 'dp-ibp':
            self.hidden2eta = nn.Sequential(nn.Linear(args.hidden_dim, args.disc), nn.Softmax(dim=-1))
            self.hidden2xia = nn.Sequential(nn.Linear(args.hidden_dim, args.disc), nn.ReLU(True))
            self.hidden2xib = nn.Sequential(nn.Linear(args.hidden_dim, args.disc), nn.ReLU(True))
            
        # decoder sub-model
        self.latent2features = nn.Sequential(nn.Linear(args.cont+args.disc, args.hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(args.hidden_dim, 64 * 4 * 4),
                                             nn.ReLU())
        decode_layers = []
        if args.img_size == 64:
            decode_layers += [nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=1),
                              nn.BatchNorm2d(64),
                              nn.ReLU(True)]
        decode_layers += [nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1),
                          nn.BatchNorm2d(32),
                          nn.ReLU(True),
                          nn.ConvTranspose2d(32, 32, (4, 4), stride=2, padding=1),
                          nn.BatchNorm2d(32),
                          nn.ReLU(True),
                          nn.ConvTranspose2d(32, args.img_channel, (4, 4), stride=2, padding=1),
                          nn.Sigmoid()]
        self.features2img = nn.Sequential(*decode_layers)
    
    def encoder(self, x):
        features = self.img2features(x.view(-1, self.args.img_channel, self.args.img_size, self.args.img_size))
        hiddens = self.features2hidden(features.view(-1, 64*4*4))
        pi = self.hidden2pi(hiddens)
        pi = pi.clamp(min=0.0001, max=0.9999)
        phi = self.hidden2phi(hiddens)
        psi = self.hidden2psi(hiddens)
        psi = torch.exp(0.5 * psi)  # std = exp(logvar * 0.5)
        taua = F.softplus(self.hidden2taua(hiddens)) + 0.1
        taub = F.softplus(self.hidden2taub(hiddens)) + 0.1
        if self.args.model_type == 'ibp':
            return pi, phi, psi, taua, taub
        elif self.args.model_type == 'c-ibp':
            eta = self.hidden2eta(hiddens)
            eta = eta.clamp(min=0.0001, max=0.9999)
            return pi, phi, psi, taua, taub, eta
        elif self.args.model_type == 'dp-ibp':
            eta = self.hidden2eta(hiddens)
            eta = eta.clamp(min=0.0001, max=0.9999)
            xia = F.softplus(self.hidden2xia(hiddens)) + 0.1
            xib = F.softplus(self.hidden2xib(hiddens)) + 0.1
            return pi, phi, psi, taua, taub, eta, xia, xib
    
    def decoder(self, z):
        features = self.latent2features(z)
        return self.features2img(features.view(-1, 64, 4, 4)).view(z.size(0), -1)
    
    def forward(self, x):
        if self.args.model_type == 'ibp':
            pi, phi, psi, taua, taub = self.encoder(x)
        elif self.args.model_type == 'c-ibp':
            pi, phi, psi, taua, taub, eta = self.encoder(x)
        elif self.args.model_type == 'dp-ibp':
            pi, phi, psi, taua, taub, eta, xia, xib = self.encoder(x)
            
        gaussian_dist = Normal(phi, psi)
        gaussian_sample = gaussian_dist.rsample()
        
        mask_dist = RelaxedBernoulli(self.args.tau, pi)
        mask = mask_dist.rsample()
        
        cont_representation = gaussian_sample * mask
        # if torch.isnan(rb_sample).any() or torch.isinf(rb_sample).any():
        #     raise RuntimeError('NaNs!!')
        if self.args.model_type != 'ibp':
            disc_dist = RelaxedOneHotCategorical(temperature=self.args.tau, probs=eta)
            disc_representation = disc_dist.rsample()
            representation = torch.cat([cont_representation, disc_representation], dim=-1)
        else:
            representation = cont_representation
        xbar = self.decoder(representation)
        if self.args.model_type == 'ibp':
            return xbar, pi, phi, psi, taua, taub
        elif self.args.model_type == 'c-ibp':
            return xbar, pi, phi, psi, taua, taub, eta
        elif self.args.model_type == 'dp-ibp':
            return xbar, pi, phi, psi, taua, taub, eta, xia, xib
