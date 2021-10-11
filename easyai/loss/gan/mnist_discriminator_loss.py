#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_GAN_D_LOSS
from easyai.utility.logger import EasyLogger


@REGISTERED_GAN_D_LOSS.register_module(LossName.MNISTDiscriminatorLoss)
class MNISTDiscriminatorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.MNISTDiscriminatorLoss)
        self.loss_function = nn.BCELoss()

    def forward(self, outputs, batch_data=None):
        if batch_data is not None:
            device = outputs.device
            targets = batch_data['label'].to(device)
            real_labels = torch.ones_like(targets, dtype=torch.float).to(device)
            fake_labels = torch.zeros_like(targets, dtype=torch.float).to(device)
            real_loss = self.loss_function(outputs[0].to(device), real_labels)
            fake_loss = self.loss_function(outputs[1].to(device), fake_labels)
            loss = real_loss + fake_loss
            EasyLogger.debug('MNIST_GAN D(x): {:.6f}, D(G(z)): {:.6f}'.format(torch.mean(outputs[0]).item(),
                                                                              torch.mean(outputs[1]).item()))
        else:
            loss = None
        return loss
