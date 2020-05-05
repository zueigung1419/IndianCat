#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:55:14 2019

@author: johanm
"""

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions import Normal, Beta, Bernoulli, Categorical
import numpy as np


def kl_normal(phi, psi, step, args):
    cap = min(args.a_cap, step * args.a_cap / args.total_steps)
    phi_prior = torch.zeros_like(phi)
    psi_prior = torch.ones_like(psi)
    gaussian_prior = Normal(phi_prior, psi_prior)
    gaussian_posterior = Normal(phi, psi)
    kla_loss = kl_divergence(gaussian_posterior, gaussian_prior).sum(dim=1).mean()
    cap_kla_loss = args.gamma_a * (kla_loss - cap).abs()
    return cap_kla_loss


def kl_bernoulli(pi, step, args):
    cap = min(args.h_cap, step * args.h_cap / args.total_steps)
    beta_dist = Beta(torch.ones_like(pi) * args.alpha_0, torch.ones_like(pi))
    pi_prior = Bernoulli(torch.cumprod(beta_dist.sample(), dim=-1))
    pi_posterior = Bernoulli(pi)
    klh_loss = kl_divergence(pi_posterior, pi_prior).sum(dim=1).mean()
    cap_klh_loss = args.gamma_h * (klh_loss - cap).abs()
    return cap_klh_loss


def kl_categorical_unifom(eta, step, args):
    cap = min(args.m_cap, step * args.m_cap / args.total_steps)
    # cap = min(cap, np.log(args.disc))
    klm_loss = torch.sum(eta * eta.log(), dim=1).mean() + np.log(args.disc)
    cap_klm_loss = args.gamma_m * (klm_loss - cap).abs()
    return cap_klm_loss


def kl_categorical_dp(eta, step, args):
    cap = min(args.m_cap, step * args.m_cap / args.total_steps)
    # cap = min(cap, np.log(args.disc))
    beta_dist = Beta(torch.ones_like(eta), torch.ones_like(eta) * args.beta_0)
    beta_sample = beta_dist.sample()
    neg_prod = torch.cumprod(1.0-beta_sample, dim=-1)
    beta_sample[:, 1:] = beta_sample[:, :-1] * neg_prod[:, :-1]
    beta_sample = F.softmax(beta_sample, dim=-1)
    cat_prior = Categorical(probs=beta_sample)
    cat_posterior = Categorical(probs=eta)
    klm_loss = kl_divergence(cat_posterior, cat_prior).mean()
    cap_klm_loss = args.gamma_m * (klm_loss - cap).abs()
    return cap_klm_loss


def kl_beta_tau(taua, taub, step, args):
    beta_prior = Beta(torch.ones_like(taua)*args.alpha_0, torch.ones_like(taub))
    beta_posterior = Beta(taua, taub)
    klv_loss = kl_divergence(beta_posterior, beta_prior).sum(dim=1).mean()
    return klv_loss


def kl_beta_xi(xia, xib, step, args):
    beta_prior = Beta(torch.ones_like(xia), torch.ones_like(xib)*args.beta_0)
    beta_posterior = Beta(xia, xib)
    klu_loss = kl_divergence(beta_posterior, beta_prior).sum(dim=1).mean()
    return klu_loss
