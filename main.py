#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:53:46 2019

@author: johanm
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from datetime import datetime
import random
import os
import sys
from os.path import join
import argparse
import json
from tqdm import tqdm

from model import Model
from utils import *
from dataset import *
from visulize import Visualizer


parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('--dataset', default='chairs', type=str, help='dataset name', choices=['mnist', 'fashion_mnist', 'chairs', 'celeba', 'dsprites'])
parser.add_argument('--model_type', default='c-ibp', type=str, help='model name', choices=['ibp', 'c-ibp', 'dp-ibp'])
parser.add_argument('--cont', default=30, type=int, help='latent continuous dimensions')
parser.add_argument('--disc', default=6, type=int, help='latent discrete dimensions')
parser.add_argument('--img_channel', default=1, type=int, help='dataset color channels')
parser.add_argument('--img_size', default=64, type=int, help='dataset image shape')
parser.add_argument('--num_epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--lr', default=5.0e-3, type=float, help='learning rate')
parser.add_argument('--hidden_dim', default=256, type=int, help='size of hidden dimension')
parser.add_argument('--alpha_0', default=10.0, type=float, help='IBP concentration hyper-parameter')
parser.add_argument('--beta_0', default=1, type=int, help='DP hyper-parameter')
parser.add_argument('--tau', default=0.67, type=float, help='Concrete distribution annealing parameter')
parser.add_argument('--save_dir', default='running_results')
parser.add_argument('--log_freq', default=10, type=int, help='num iterations per log')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--total_steps', default=8000, type=int, help='gradually increasing steps')
parser.add_argument('--a_cap', default=15.0, type=float, help='channel capacity of latent gaussian variable a')
parser.add_argument('--h_cap', default=15.0, type=float, help='channel capacity of latent bernoulli variable h')
parser.add_argument('--m_cap', default=15.0, type=float, help='channel capacity of latent discrete variable m')
parser.add_argument('--gamma_a', default=30, type=float, help='penalty coefficient of latent gaussian variable a')
parser.add_argument('--gamma_h', default=30, type=float, help='penalty coefficient of latent bernoulli variable h')
parser.add_argument('--gamma_m', default=30, type=float, help='penalty coefficient of latent discrete variable a')
parser.add_argument('--verbose', default=True, type=bool, help='print more help information')
args = parser.parse_args()
if args.verbose:
    json.dump(args.__dict__, sys.stdout, indent=4)

# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# np.random.seed(args.seed)
# random.seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
running_dir = join(args.dataset, args.model_type, date_time)
if not os.path.exists(running_dir):
    os.makedirs(running_dir)
with open(join(running_dir, 'config.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

if args.dataset == 'mnist':
    train_loader, test_loader = get_mnist_dataloader(batch_size=args.batch_size)
elif args.dataset == 'fahsion_mnist':
    train_loader, test_loader = get_fashion_mnist_dataloader(batch_size=args.batch_size)
elif args.dataset == 'chairs':
    train_loader = get_chairs_dataloader(batch_size=args.batch_size)
elif args.dataset == 'celeba':
    train_loader = get_celeba_dataloader(batch_size=args.batch_size)
elif args.dataset == 'dsprites':
    train_loader = get_dsprites_dataloader(batch_size=args.batch_size)
else:
    raise NotImplementedError('Not implemented for dataset {}'.format(args.dataset))

if not torch.cuda.is_available():
    raise RuntimeError('No cuda device')
model = Model(args).cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
viz = Visualizer(model, args)

step = 0
start_time = time.time()
for epoch in range(1, args.num_epochs+1):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (x, _) in pbar:
        step += 1
        x = x.cuda()
        optimizer.zero_grad()
        if args.model_type == 'ibp':
            xbar, pi, phi, psi, taua, taub = model(x)
        elif args.model_type == 'c-ibp':
            xbar, pi, phi, psi, taua, taub, eta = model(x)
        elif args.model_type == 'dp-ibp':
            xbar, pi, phi, psi, taua, taub, eta, xia, xib = model(x)
        recon_loss = F.binary_cross_entropy(xbar.view(xbar.size(0), -1), x.view(x.size(0), -1)) * args.img_channel * args.img_size * args.img_size
        cap_kla_loss = kl_normal(phi, psi, step, args)
        cap_klh_loss = kl_bernoulli(pi, step, args)
        klv_loss = kl_beta_tau(taua, taub, step, args)
        loss = recon_loss + cap_kla_loss + cap_klh_loss + klv_loss
        if args.model_type == 'c-ibp':
            klm_loss = kl_categorical_unifom(eta, step, args)
            loss += klm_loss
        if args.model_type == 'dp-ibp':
            klm_loss = kl_categorical_dp(eta, step, args)
            loss += klm_loss
            klu_loss = kl_beta_xi(xia, xib, step, args)
            loss += klu_loss
        loss.backward()
        optimizer.step()
        pbar.set_description('model_type: {}, epoch: {}, loss: {:.2f}'.format(args.model_type, epoch, loss.item()))
    if epoch % 10 == 0:
        for i in range(10):
            # viz.traverse_line(cont_dim=0, disc_dim=1, file_name=join(running_dir, f'epoch_{epoch}_0.jpeg'))
            viz.traverse_grid(cont_dim=i, nrow=10, ncol=10, use_prior=False, set_zero=True, file_name=join(running_dir, f'traversal_dim{i}.jpeg'))
torch.save(model.state_dict(), join(running_dir, 'model.pt'))
print('time elapsed: {:.2f} minutes'.format((time.time()-start_time) / 60.0))








