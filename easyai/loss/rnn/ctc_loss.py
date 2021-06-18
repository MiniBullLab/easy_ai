#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS


@REGISTERED_RNN_LOSS.register_module(LossName.CTCLoss)
class CTCLoss(BaseLoss):

    def __init__(self, blank_index, reduction='mean'):
        super().__init__(LossName.CTCLoss)
        self.loss_func = torch.nn.CTCLoss(blank=blank_index,
                                          reduction=reduction)

    def forward(self, input_data, target_dict=None):
        if target_dict is not None:
            batch_size = input_data.size(0)
            label, label_length = target_dict['targets'], target_dict['targets_lengths']
            pred = input_data.log_softmax(2)
            pred = pred.permute(1, 0, 2)
            preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
            loss = self.loss_func(pred, label, preds_lengths, label_length)
        else:
            loss = F.softmax(input_data, dim=2)
        return loss
