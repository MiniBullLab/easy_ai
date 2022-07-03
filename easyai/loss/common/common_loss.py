#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_COMMON_LOSS


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


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

    def forward(self, input_data, batch_data=None):
        if batch_data is not None:
            return torch.Tensor([0])
        else:
            return input_data
