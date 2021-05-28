#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.config.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.cls.ce2d_loss import CrossEntropy2dLoss
from easyai.loss.cls.ce2d_loss import BinaryCrossEntropy2dLoss
from easyai.loss.utility.registry import REGISTERED_SEG_LOSS


@REGISTERED_SEG_LOSS.register_module(LossName.MixCrossEntropy2dLoss)
class MixCrossEntropy2dLoss(BaseLoss):

    def __init__(self, aux_weight=0.2, weight_type=0, weight=None,
                 reduction='mean', ignore_index=250):
        super().__init__(LossName.MixCrossEntropy2dLoss)
        self.aux_weight = aux_weight
        self.ce2d_loss = CrossEntropy2dLoss(weight_type, weight,
                                            reduction, ignore_index)

    def _aux_loss(self, inputs, target):
        loss = self.ce2d_loss(inputs[0], target)
        for i in range(1, len(inputs)):
            aux_loss = self.ce2d_loss(inputs[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self._aux_loss(input_data, target)
        else:
            loss = F.softmax(input_data[0], dim=1)
        return loss


@REGISTERED_SEG_LOSS.register_module(LossName.MixBinaryCrossEntropy2dLoss)
class MixBinaryCrossEntropy2dLoss(BaseLoss):

    def __init__(self, aux_weight=0.2, weight_type=0, weight=None,
                 reduction='mean', ignore_index=250):
        super().__init__(LossName.MixBinaryCrossEntropy2dLoss)
        self.aux_weight = aux_weight
        self.bce_loss = BinaryCrossEntropy2dLoss(weight_type, weight,
                                                 reduction, ignore_index)

    def _aux_loss(self, inputs, target):
        loss = self.bce_loss(inputs[0], target)
        for i in range(1, len(inputs)):
            aux_loss = self.bce_loss(inputs[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, input_data, target=None):
        if target is not None:
            loss = self._aux_loss(input_data, target)
        else:
            loss = input_data[0]
        return loss
