#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS
from easyai.utility.logger import EasyLogger


@REGISTERED_RNN_LOSS.register_module(LossName.CTCLoss)
class CTCLoss(BaseLoss):

    def __init__(self, blank_index, reduction='mean'):
        super().__init__(LossName.CTCLoss)
        self.blank_index = blank_index
        self.loss_func = torch.nn.CTCLoss(blank=blank_index,
                                          reduction=reduction)

    def forward(self, input_data, batch_data=None):
        if batch_data is not None:
            batch_size = input_data.size(0)
            device = input_data.device
            pred = input_data.log_softmax(2)
            pred = pred.permute(1, 0, 2)  # T * N * C
            seq_len = pred.size(0)
            targets = torch.full(size=(batch_size, seq_len),
                                 fill_value=self.blank_index,
                                 dtype=torch.long, device=device)
            for idx, tensor in enumerate(batch_data['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]

            target_lengths = torch.clamp(batch_data['targets_lengths'],
                                         min=1, max=seq_len).long().to(device)

            input_lengths = torch.full(size=(batch_size,),
                                       fill_value=seq_len,
                                       dtype=torch.long,
                                       device=device)
            loss = self.loss_func(pred, targets,
                                  input_lengths, target_lengths)
            if loss.item() == float("inf"):
                EasyLogger.error("{} {} {}".format(batch_data['label'],
                                                   pred.shape, target_lengths))
        else:
            loss = F.softmax(input_data, dim=2)
        return loss
