#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
Bag of Tricks for Image Classification with Convolutional Neural Networks
"""
import torch
from easyai.name_manager.solver_name import OptimizerName
from easyai.solver.utility.registry import REGISTERED_OPTIMIZER


@REGISTERED_OPTIMIZER.register_module(OptimizerName.LARCOptimizer)
class LARCOptimizer():
    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optimizer = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optimizer.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (
                                    grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optimizer.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optimizer.param_groups):
            group['weight_decay'] = weight_decays[i]

    def zero_grad(self):
        self.optimizer.zero_grad()
