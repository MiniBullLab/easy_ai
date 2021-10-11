#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_GAN_G_LOSS


@REGISTERED_GAN_G_LOSS.register_module(LossName.MNISTGeneratorLoss)
class MNISTGeneratorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.MNISTGeneratorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, batch_data=None):
        if batch_data is not None:
            device = outputs.device
            targets = batch_data['label'].to(device)
            real_labels = torch.ones_like(targets, dtype=torch.float).to(device)
            loss = self.loss_function(outputs.to(device), real_labels)
        else:
            loss = outputs
        return loss
