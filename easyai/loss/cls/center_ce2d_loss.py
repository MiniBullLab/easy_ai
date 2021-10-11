#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.common.center_loss import CenterLoss
from easyai.loss.utility.loss_registry import REGISTERED_CLS_LOSS


@REGISTERED_CLS_LOSS.register_module(LossName.CenterCrossEntropy2dLoss)
class CenterCrossEntropy2dLoss(BaseLoss):
    def __init__(self, class_number, feature_dim=2, alpha=1,
                 reduction='mean', ignore_index=250):
        super().__init__(LossName.CenterCrossEntropy2dLoss)
        self.lr_center = 0.5
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction=self.reduction,
                                           ignore_index=self.ignore_index)
        self.center_loss = CenterLoss(class_number, feature_dim)

    def forward(self, input_data, batch_data=None):
        input_data = input_data.float()
        if batch_data is not None:
            device = input_data[0].device
            targets = batch_data['label'].to(device)
            loss1 = self.ce_loss(input_data[1], targets)
            loss2 = self.center_loss(input_data[0], targets)
            loss = loss1 + loss2 * self.alpha
        else:
            loss = F.softmax(input_data, dim=1)
        return loss
