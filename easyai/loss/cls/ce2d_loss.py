#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.calculate_weights import numpy_compute_weight
from easyai.loss.utility.registry import REGISTERED_CLS_LOSS


@REGISTERED_CLS_LOSS.register_module(LossName.CrossEntropy2dLoss)
class CrossEntropy2dLoss(BaseLoss):
    def __init__(self, weight_type=0, weight=None,
                 reduction='mean', ignore_index=250):
        super().__init__(LossName.CrossEntropy2dLoss)
        self.weight_type = weight_type
        if weight is not None:
            self.weight = torch.FloatTensor(weight)
        else:
            self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def compute_loss_from_weight(self, loss, target):
        ignore = target == self.ignore_index
        valid_count = (ignore == 0).sum()
        if self.weight_type == 1:
            result = 0
            class_number = len(self.weight)
            self.weight = self.weight.type(loss.dtype)
            for index in range(class_number):
                result += self.weight[index] * target.eq(index).type(loss.dtype) * loss
        elif self.weight_type == 2:
            result = 0
            labels = target.data.cpu().numpy()
            weights = numpy_compute_weight(labels, self.ignore_index)
            class_number = len(weights)
            for index in range(class_number):
                result += weights[index] * target.eq(index).type(loss.dtype) * loss
        else:
            result = loss
        result = target.ne(self.ignore_index).type(loss.dtype) * result
        if self.reduction == 'mean':
            return result.sum() / valid_count
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result

    def forward(self, input_data, target=None):
        input_data = input_data.float()
        if target is not None:
            if self.weight_type == 0:
                loss = F.cross_entropy(input_data, target,
                                       weight=self.weight,
                                       reduction=self.reduction,
                                       ignore_index=self.ignore_index)
            else:
                loss = F.cross_entropy(input_data, target,
                                       reduction='none',
                                       ignore_index=self.ignore_index)
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = F.softmax(input_data, dim=1)
        return loss


@REGISTERED_CLS_LOSS.register_module(LossName.BinaryCrossEntropy2dLoss)
class BinaryCrossEntropy2dLoss(BaseLoss):

    def __init__(self, weight_type=0, weight=None,
                 reduction='mean', ignore_index=250):
        super().__init__(LossName.BinaryCrossEntropy2dLoss)
        self.weight_type = weight_type
        if weight is not None:
            self.weight = torch.FloatTensor(weight)
        else:
            self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def compute_loss_from_weight(self, loss, target):
        ignore = target == self.ignore_index
        valid_count = (ignore == 0).sum()

        self.weight = self.weight.type(loss.dtype)
        if self.weight_type == 1:
            result = self.weight[0] * target.eq(0).type(loss.dtype) * loss + \
                     self.weight[1] * target.eq(1).type(loss.dtype) * loss
        elif self.weight_type == 2:
            labels = target.data.cpu().numpy()
            weights = numpy_compute_weight(labels, self.ignore_index)
            result = weights[0] * target.eq(0).type(loss.dtype) * loss + \
                     weights[1] * target.eq(1).type(loss.dtype) * loss
        else:
            result = loss
        result = target.ne(self.ignore_index).type(loss.dtype) * result
        if self.reduction == 'mean':
            return result.sum() / valid_count
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result

    def forward(self, input_data, target=None):
        if target is not None:
            target = target.type(input_data.dtype)
            if self.weight_type == 0:
                loss = F.binary_cross_entropy(input_data, target,
                                              weight=self.weight,
                                              reduction=self.reduction)
            else:
                loss = F.binary_cross_entropy(input_data, target,
                                              reduction='none')
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = input_data
        return loss
