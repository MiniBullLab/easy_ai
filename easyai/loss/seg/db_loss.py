#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.cls.ce2d_loss import CrossEntropy2dLoss
from easyai.loss.cls.ce2d_loss import BinaryCrossEntropy2dLoss
from easyai.loss.utility.loss_registry import REGISTERED_SEG_LOSS


@REGISTERED_SEG_LOSS.register_module(LossName.DBLoss)
class DBLoss(BaseLoss):

    def __init__(self, reduction='mean', ignore_index=250):
        super().__init__(LossName.DBLoss)

    def forward(self, input_data, target=None):
        input_data = input_data.float()
        if target is not None:
            loss = 0
        else:
            loss = input_data
        return loss
