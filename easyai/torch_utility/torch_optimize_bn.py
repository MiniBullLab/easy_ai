#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import torch.nn as nn
from easyai.model_block.base_block.common.utility_layer import EmptyLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.utility.logger import EasyLogger


class TorchOptimizeBN():

    def __init__(self, optimize_type=0):
        self.optimize_type = optimize_type

    def update_front_layer(self, model, sr_lr, layer_name):
        for key, block in model._modules.items():
            self.update_bn(block, sr_lr)
            if layer_name == key:
                break

    def update_bn(self, model, sr_lr):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m.weight, 'data'):
                    m.weight.grad.data.add_(sr_lr * torch.sign(m.weight.data))

    def fuse_bn_recursively(self, model):
        for module_name in model._modules:
            model._modules[module_name] = self.fuse_bn_sequential(model._modules[module_name])
            if len(model._modules[module_name]._modules) > 0:
                self.fuse_bn_recursively(model._modules[module_name])
        return model

    def fuse_bn_sequential(self, block):
        """
        This function takes a sequential block and fuses the batch normalization with convolution
        :param model: nn.Sequential. Source resnet model
        :return: nn.Sequential. Converted block
        """
        if not isinstance(block, nn.Sequential):
            return block
        stack = []
        for m in block.children():
            if isinstance(m, nn.BatchNorm2d):
                if isinstance(stack[-1], nn.Conv2d):
                    bn_st_dict = m.state_dict()
                    conv_st_dict = stack[-1].state_dict()

                    # BatchNorm params
                    eps = m.eps
                    mu = bn_st_dict['running_mean']
                    var = bn_st_dict['running_var']
                    gamma = bn_st_dict['weight']

                    if 'bias' in bn_st_dict:
                        beta = bn_st_dict['bias']
                    else:
                        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                    # Conv params
                    W = conv_st_dict['weight']
                    if 'bias' in conv_st_dict:
                        bias = conv_st_dict['bias']
                    else:
                        bias = torch.zeros(W.size(0)).float().to(gamma.device)

                    denom = torch.sqrt(var + eps)
                    b = beta - gamma.mul(mu).div(denom)
                    A = gamma.div(denom)
                    bias *= A
                    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                    W.mul_(A)
                    bias.add_(b)

                    stack[-1].weight.data.copy_(W)
                    if stack[-1].bias is None:
                        stack[-1].bias = torch.nn.Parameter(bias)
                    else:
                        stack[-1].bias.data.copy_(bias)

            else:
                stack.append(m)

        if len(stack) > 1:
            return nn.Sequential(*stack)
        else:
            return stack[0]

    def fuse(self, model):  # fuse model Conv2d() + BatchNorm2d() layers
        EasyLogger.info('Fusing layers... ')
        for block in model.modules():
            if isinstance(block, (ConvBNActivationBlock,)):
                block.block[0] = self.fuse_conv_and_bn(block.block[0],
                                                       block.block[1])  # update conv
                block.block[1] = EmptyLayer()  # remove batchnorm

    def fuse_conv_and_bn(self, conv, bn):
        # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              groups=conv.groups,
                              bias=True).requires_grad_(False).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

