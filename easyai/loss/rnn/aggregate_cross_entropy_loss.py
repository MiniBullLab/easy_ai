#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS
from easyai.utility.logger import EasyLogger


@REGISTERED_RNN_LOSS.register_module(LossName.AggregationCrossEntropyLoss)
class AggregationCrossEntropyLoss(BaseLoss):

    def __init__(self, class_number):
        super().__init__(LossName.AggregationCrossEntropyLoss)
        self.class_number = class_number

    def forward(self, input_data, batch_data=None):
        """
            :param input_data: [B,T,C]
            :param batch_data['targets']: [B,C]
            :return loss:
        """
        if batch_data is not None:
            batch_size = input_data.size(0)
            device = input_data.device
            pred = F.softmax(input_data, dim=2)
            pred = pred + 1e-10
            seq_len = pred.size(1)
            label = batch_data['targets'].to(device)
            label[:, 0] = seq_len - label[:, 0]

            # ACE Implementation (four fundamental formulas)
            pred = torch.sum(pred, 1)
            pred = pred / seq_len
            label = label / seq_len
            loss = (-torch.sum(torch.log(pred) * label)) / batch_size
        else:
            loss = F.softmax(input_data, dim=2)
        return loss
