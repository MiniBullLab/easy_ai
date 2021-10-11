#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_GAN_D_LOSS


@REGISTERED_GAN_D_LOSS.register_module(LossName.GANomalyDiscriminatorLoss)
class GANomalyDiscriminatorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.GANomalyDiscriminatorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, batch_data=None):
        if batch_data is not None:
            device = outputs[0].device
            targets = batch_data['label'].to(device)
            real_data = outputs[0].view(-1, 1).squeeze(1)
            fake_data = outputs[1].view(-1, 1).squeeze(1)
            real_labels = torch.ones_like(targets, dtype=torch.float).to(targets.device)
            fake_labels = torch.zeros_like(targets, dtype=torch.float).to(targets.device)
            real_loss = self.loss_function(real_data.to(targets.device), real_labels)
            fake_loss = self.loss_function(fake_data.to(targets.device), fake_labels)
            loss = (real_loss + fake_loss) * 0.5
        else:
            loss = None
        return loss
