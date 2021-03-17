#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_POSE2D_LOSS


@REGISTERED_POSE2D_LOSS.register_module(LossName.JointsMSELoss)
class JointsMSELoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.JointsMSELoss)
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = True

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (Tensor))
            targets (Tensor)

        Returns:
            loss (Tensor)
        """
        pass

