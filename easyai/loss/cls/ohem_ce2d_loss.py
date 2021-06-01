#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_CLS_LOSS


@REGISTERED_CLS_LOSS.register_module(LossName.OhemCrossEntropy2dLoss)
class OhemCrossEntropy2dLoss(BaseLoss):

    def __init__(self, threshold=0.7, min_keep=1, ignore_index=250):
        super().__init__(LossName.OhemCrossEntropy2dLoss)
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_keep = min_keep
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean',
                                                       ignore_index=ignore_index)

    def get_hard_example(self, input_data, target):
        ignore = target == self.ignore_index
        valid_count = (ignore == 0).sum()
        if valid_count <= 0 or self.min_keep > valid_count:
            print("OHEM Lables: {}".format(valid_count))
            return target

        c = input_data.shape[1]
        target_shape = target.shape
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()

        prob = F.softmax(input_data, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        prob = prob.masked_fill_(1 - valid_mask, 1)
        mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
        threshold = self.threshold
        if self.min_keep > 0:
            # index = mask_prob.argsort()
            index = np.argsort(mask_prob.cpu().detach().numpy())
            threshold_index = index[min(len(index), self.min_keep) - 1]
            if mask_prob[threshold_index] > self.threshold:
                threshold = mask_prob[threshold_index]
        keep_mask = mask_prob.le(threshold)
        valid_mask = valid_mask * keep_mask
        target = target * keep_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(target_shape)
        return target

    def forward(self, input_data, target):
        if target is not None:
            target = self.get_hard_example(input_data, target)
            loss = self.loss_function(input_data, target)
        else:
            loss = F.softmax(input_data, dim=1)
        return loss


@REGISTERED_CLS_LOSS.register_module(LossName.OhemBinaryCrossEntropy2dLoss)
class OhemBinaryCrossEntropy2dLoss(BaseLoss):

    def __init__(self, threshold=0.7, min_keep=1, ignore_index=250):
        super().__init__(LossName.OhemBinaryCrossEntropy2dLoss)
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_keep = min_keep

    def get_hard_example(self, input_data, target):
        ignore = target == self.ignore_index
        valid_count = (ignore == 0).sum()
        if valid_count <= 0 or self.min_keep > valid_count:
            print("OHEM Lables: {}".format(valid_count))
            return target, valid_count

        target_shape = target.shape
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()

        prob = input_data.view(-1)
        prob = prob.masked_fill_(1 - valid_mask, 1)

        threshold = self.threshold
        if self.min_keep > 0:
            # index = mask_prob.argsort()
            index = np.argsort(prob.cpu().detach().numpy())
            threshold_index = index[min(len(index), self.min_keep) - 1]
            if prob[threshold_index] > self.threshold:
                threshold = prob[threshold_index]
        keep_mask = prob.le(threshold)
        valid_mask = valid_mask * keep_mask
        target = target * keep_mask.long()

        target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.view(target_shape)
        return target, valid_count

    def forward(self, input_data, target):
        if target is not None:
            target, valid_count = self.get_hard_example(input_data, target)
            loss = F.binary_cross_entropy(input_data, target,
                                          reduction='none')
            result = target.ne(self.ignore_index).type(loss.dtype) * loss
            loss = result.sum() / valid_count
        else:
            loss = input_data
        return loss
