#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_GAN_G_LOSS


@REGISTERED_GAN_G_LOSS.register_module(LossName.MNISTGeneratorLoss)
class MNISTGeneratorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.MNISTGeneratorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, targets):
        real_labels = torch.ones_like(targets, dtype=torch.float).to(targets.device)
        G_loss = self.loss_function(outputs[0], real_labels)
        return G_loss
