#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS
from easyai.utility.logger import EasyLogger


@REGISTERED_RNN_LOSS.register_module(LossName.CTCLoss)
class CTCLoss(BaseLoss):

    def __init__(self, blank_index, reduction='mean',
                 use_weight=False, use_focal=False,
                 alpha=0.99, gamma=1):
        super().__init__(LossName.CTCLoss)
        self.blank_index = blank_index
        self.reduction = reduction
        self.use_weight = use_weight
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = torch.nn.CTCLoss(blank=blank_index,
                                          reduction=reduction)

    def compute_weight(self, preds, target_lengths, device):
        len_index = torch.softmax(preds, -1).max(2)[1].transpose(0, 1) > 0
        len_flag = torch.cat([target_lengths.long().unsqueeze(0),
                              len_index.sum(1).unsqueeze(0)], 0)
        ctc_loss_weight = len_flag.max(0)[0].float() / len_flag.min(0)[0].float()
        ctc_loss_weight[ctc_loss_weight == torch.tensor(np.inf).to(device)] = 2.0
        return ctc_loss_weight

    def forward(self, input_data, batch_data=None):
        if batch_data is not None:
            batch_size = input_data.size(0)
            device = input_data.device
            pred = input_data.permute(1, 0, 2)  # T * N * C
            seq_len = pred.size(0)
            pred = pred.log_softmax(2).requires_grad_()
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
            ctc_loss = self.loss_func(pred, targets,
                                      input_lengths, target_lengths)
            if self.use_weight and self.reduction == "none":
                ctc_loss_weight = self.compute_weight(pred, target_lengths, device)
                loss = ctc_loss * ctc_loss_weight
                loss = loss.sum() / batch_size
            elif self.use_focal and self.reduction == "none":
                prob = torch.exp(-ctc_loss)
                focal_loss = self.alpha * (1 - prob).pow(self.gamma) * ctc_loss
                loss = focal_loss.sum() / batch_size
            else:
                loss = ctc_loss
            if loss.item() == float("inf") or loss.item() == float("nan"):
                EasyLogger.error("{} {} {}".format(batch_data['label'],
                                                   pred.shape, target_lengths))
        else:
            loss = F.softmax(input_data, dim=2)
        return loss
