#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.label_smoothing import LabelSmoothing
from easyai.loss.utility.loss_registry import REGISTERED_CLS_LOSS


# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, smoothing=0.1):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         assert smoothing < 1.0
#         self.smoothing = smoothing
#         self.confidence = 1. - smoothing
#
#     def forward(self, x, target):
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()


@REGISTERED_CLS_LOSS.register_module(LossName.LabelSmoothCE2dLoss)
class LabelSmoothCE2dLoss(BaseLoss):

    def __init__(self, class_number, epsilon=0.1, reduction='mean', ignore_index=250):
        super().__init__(LossName.LabelSmoothCE2dLoss)
        self.class_number = class_number
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            log_preds = F.log_softmax(outputs, dim=1)
            if self.reduction == 'sum':
                log_loss = -log_preds.sum()
            else:
                log_loss = -log_preds.sum(dim=-1)
                if self.reduction == 'mean':
                    log_loss = log_loss.mean()
            nll_loss = F.nll_loss(log_preds, targets, reduction=self.reduction,
                                  ignore_index=self.ignore_index)
            loss = log_loss * self.epsilon / self.class_number + (1 - self.epsilon) * nll_loss
        return loss


# @REGISTERED_CLS_LOSS.register_module(LossName.LabelSmoothCE2dLoss)
class LabelSmoothCE2dLossV2(BaseLoss):

    def __init__(self, class_number, epsilon=0.1, reduction='mean', ignore_index=250):
        super().__init__(LossName.LabelSmoothCE2dLoss)
        self.class_number = class_number
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.label_smoothing = LabelSmoothing(class_number, epsilon, ignore_index)

    def forward(self, outputs, targets=None):
        outputs = outputs.float()
        if targets is None:
            loss = F.softmax(outputs, dim=1)
        else:
            ignore = targets == self.ignore_index
            valid_count = (ignore == 0).sum()
            smooth_onehot = self.label_smoothing.smoothing(outputs, targets)
            logs = self.log_softmax(outputs)
            loss = -torch.sum(logs * smooth_onehot, dim=1)
            loss[ignore] = 0
            if self.reduction == 'mean':
                loss = loss.sum() / valid_count
            if self.reduction == 'sum':
                loss = loss.sum()
        return loss
