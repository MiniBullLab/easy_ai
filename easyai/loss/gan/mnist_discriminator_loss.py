#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.loss.utility.base_loss import *


class MNISTDiscriminatorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossType.MNISTDiscriminatorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, targets):
        real_labels = torch.ones_like(targets, dtype=torch.float).to(targets.device)
        fake_labels = torch.zeros_like(targets, dtype=torch.float).to(targets.device)
        D_real_loss = self.loss_function(outputs[0], real_labels)
        D_fake_loss = self.loss_function(outputs[0], fake_labels)
        loss = D_real_loss + D_fake_loss
        return loss
