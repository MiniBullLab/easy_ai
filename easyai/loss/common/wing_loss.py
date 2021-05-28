#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.config.name_manager import LossName
from easyai.loss.utility.base_loss import *


class WingLoss(BaseLoss):

    def __init__(self, w=10, epsilon=2, weight=None, ignore_value=-1):
        super().__init__(LossName.WingLoss)
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
        self.weight = weight
        self.ignore_value = ignore_value

    def forward(self, output, targets):
        device = output.device
        batch_size = output.size(0)
        predictions = output.reshape((batch_size, -1))
        gt_data = targets.reshape((batch_size, -1)).to(device)
        # if gt_data == -1
        predictions = torch.where(gt_data == self.ignore_value, gt_data, predictions)
        # common loss
        x = predictions - gt_data
        if self.weight is not None:
            x = x * self.weight
        t = torch.abs(x)
        wing_loss = torch.where(t < self.w, self.w * torch.log(1 + t / self.epsilon), t - self.C)
        return torch.mean(wing_loss)
