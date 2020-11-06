#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.calculate_weights import numpy_compute_weight
from easyai.loss.utility.registry import REGISTERED_CLS_LOSS


@REGISTERED_CLS_LOSS.register_module(LossName.CrossEntropy2d)
class CrossEntropy2d(BaseLoss):
    def __init__(self, weight_type=0, weight=None,
                 reduce=None, reduction='mean', ignore_index=250):
        super().__init__(LossName.CrossEntropy2d)
        self.weight_type = weight_type
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        if weight_type == 0:
            self.loss_function = nn.CrossEntropyLoss(weight=self.weight,
                                                     reduce=self.reduce,
                                                     reduction=self.reduction,
                                                     ignore_index=self.ignore_index)
        else:
            self.loss_function = nn.CrossEntropyLoss(reduce=self.reduce,
                                                     reduction=self.reduction,
                                                     ignore_index=self.ignore_index)

    def compute_loss_from_weight(self, loss, target):
        weight = [float(x) for x in self.weight.split(',') if x]
        return loss

    def forward(self, outputs, target=None):
        outputs = outputs.float()
        if target is not None:
            loss = self.loss_function(outputs, target)
            if self.weight_type != 0 and self.weight is not None:
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = F.softmax(outputs, dim=1)
        return loss


@REGISTERED_CLS_LOSS.register_module(LossName.BinaryCrossEntropy2d)
class BinaryCrossEntropy2d(BaseLoss):

    def __init__(self, weight_type=0, weight=None,
                 reduce=None, reduction='mean', ignore_index=250):
        super().__init__(LossName.BinaryCrossEntropy2d)
        self.weight_type = weight_type
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        if weight_type == 0:
            self.loss_function = torch.nn.BCELoss(weight=self.weight, reduce=self.reduce,
                                                  reduction=self.reduction)
        else:
            self.loss_function = torch.nn.BCELoss(reduce=False, reduction=self.reduction)

    def compute_loss_from_weight(self, loss, target):
        ignore = target == self.ignore_index
        valid_count = (ignore == 0).sum()
        if self.weight_type == 1:
            weights = [float(x) for x in self.weight.split(',') if x]
            result = weights[0] * target.eq(0).type(loss.dtype) * loss + \
                     weights[1] * target.eq(1).type(loss.dtype) * loss
        elif self.weight_type == 2:
            labels = target.data.cpu().numpy()
            weights = numpy_compute_weight(labels)
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
            loss = self.loss_function(input_data, target)
            if self.weight_type != 0:
                loss = self.compute_loss_from_weight(loss, target)
        else:
            loss = input_data
        return loss
