#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_GAN_D_LOSS


@REGISTERED_GAN_D_LOSS.register_module(LossName.MNISTDiscriminatorLoss)
class MNISTDiscriminatorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.MNISTDiscriminatorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, targets=None):
        if targets is not None:
            real_labels = torch.ones_like(targets, dtype=torch.float).to(targets.device)
            fake_labels = torch.zeros_like(targets, dtype=torch.float).to(targets.device)
            real_loss = self.loss_function(outputs[0], real_labels)
            fake_loss = self.loss_function(outputs[1], fake_labels)
            loss = real_loss + fake_loss
        else:
            loss = None
        return loss
