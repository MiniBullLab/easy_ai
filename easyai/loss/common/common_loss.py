#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_COMMON_LOSS


def smooth_l1_loss(x, t):
    diff = (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < 1.).float()
    y = flag * (diff ** 2) * 0.5 + (1 - flag) * (abs_diff - 0.5)
    return y.sum()


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

