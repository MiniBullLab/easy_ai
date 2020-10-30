#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


class MNISTGeneratorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossType.MNISTGeneratorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, targets):
        real_labels = torch.ones_like(targets, dtype=torch.float).to(targets.device)
        G_loss = self.loss_function(outputs[0], real_labels)
        return G_loss
