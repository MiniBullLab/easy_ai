#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_COMMON_LOSS


def smooth_l1_loss(x, t):
    diff = (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < 1.).float()
    y = flag * (diff ** 2) * 0.5 + (1 - flag) * (abs_diff - 0.5)
    return y.sum()


def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)


class GaussianNLLoss(nn.Module):
    def __init__(self, scale=1., reduction='mean',
                 ignore_value=-100):
        super().__init__()
        self.scale = scale
        self.reduction = reduction
        self.ignore_value = ignore_value

    def gaussian_dist_pdf(self, val, mean, sigma, scale, sigma_const=0.3):
        pi = torch.tensor(np.pi)
        Z = torch.sqrt(2.0 * pi) * (sigma + sigma_const)
        return torch.exp(-0.5 * ((val - mean) / scale) ** 2.0 / ((sigma + sigma_const) ** 2)) / Z

    def forward(self, outputs, target):
        prediction = outputs[0]
        out_gauss = outputs[1]
        target = target.type(prediction.dtype)
        loss_weight = torch.where(target.clone() == self.ignore_value,
                                  torch.zeros_like(target.clone()),
                                  torch.ones_like(target.clone())).detach()

        loss = -torch.log(self.gaussian_dist_pdf(prediction, target,
                                                 out_gauss, self.scale) + 1e-9)

        loss = loss * loss_weight
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


@REGISTERED_COMMON_LOSS.register_module(LossName.EmptyLoss)
class EmptyLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.EmptyLoss)

    def forward(self, input_data, target=None):
        pass


@REGISTERED_COMMON_LOSS.register_module(LossName.MeanSquaredErrorLoss)
class MeanSquaredErrorLoss(BaseLoss):

    def __init__(self, reduction='mean'):
        super().__init__(LossName.MeanSquaredErrorLoss)
        self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self.loss_function(input_data, target)
        else:
            loss = input_data
        return loss

