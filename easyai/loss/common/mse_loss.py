# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_COMMON_LOSS


@REGISTERED_COMMON_LOSS.register_module(LossName.MeanSquaredErrorLoss)
class MeanSquaredErrorLoss(BaseLoss):

    def __init__(self, reduction='mean'):
        super().__init__(LossName.MeanSquaredErrorLoss)
        self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def forward(self, input_data, batch_data=None):
        if batch_data is not None:
            device = input_data.device
            targets = batch_data['label'].to(device)
            loss = self.loss_function(input_data, targets)
        else:
            loss = input_data
        return loss



