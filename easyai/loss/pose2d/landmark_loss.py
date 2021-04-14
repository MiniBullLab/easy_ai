#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_POSE2D_LOSS


@REGISTERED_POSE2D_LOSS.register_module(LossName.LandmarkLoss)
class LandmarkLoss(BaseLoss):

    def __init__(self, points_count):
        super().__init__(LossName.LandmarkLoss)
        self.points_count = points_count

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (Tensor))
            targets (Tensor, Tensor)

        Returns:
            loss (Tensor)
        """
        ldmk = outputs[0]
        conf = outputs[1]
        gauss = outputs[2]
        if targets is None:
            final_gauss = torch.zeros_like(conf, dtype=torch.float).to(conf.device)
            for i in range(self.points_count):
                final_gauss[:, i] = 1.0 - 0.5 * (gauss[:, 2 * i] + gauss[:, 2 * i + 1])
            landmark_conf = conf * final_gauss
            return ldmk, landmark_conf
        else:
            pass
