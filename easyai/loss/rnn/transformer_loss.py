#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS


@REGISTERED_RNN_LOSS.register_module(LossName.TransformerLoss)
class TransformerLoss(BaseLoss):

    def __init__(self, ignore_index=2, reduction='mean'):
        super().__init__(LossName.TransformerLoss)
        self.ignore_index = ignore_index
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index,
                                                   reduction=reduction)

    def forward(self, input_data, batch_data=None):
        if batch_data is not None:
            device = input_data.device
            targets = batch_data['targets'][:, 1:].to(device)  # without [GO] Symbol
            loss = self.loss_func(input_data.view(-1, input_data.shape[-1]),
                                  targets.contiguous().view(-1))
        else:
            loss = F.softmax(input_data, dim=2)
        return loss
