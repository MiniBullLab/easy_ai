#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.common.common_loss import l2_loss
from easyai.loss.utility.registry import REGISTERED_GAN_G_LOSS


@REGISTERED_GAN_G_LOSS.register_module(LossName.GANomalyGeneratorLoss)
class GANomalyGeneratorLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.GANomalyGeneratorLoss)
        self.adv_weight = 1  # Adversarial loss weight
        self.con_weight = 50  # Reconstruction loss weight
        self.enc_weight = 1  # Encoder loss weight
        self.adv_loss = l2_loss
        self.con_loss = nn.L1Loss()
        self.enc_loss = l2_loss

        self.loss_info = {'adv_loss': 0, 'con_loss': 0,
                          'enc_loss': 0}

    def forward(self, outputs, targets=None):
        if targets is not None:
            adv_error = self.adv_loss(outputs[-2], outputs[-1])
            con_error = self.con_loss(outputs[2], outputs[0])
            enc_error = self.enc_loss(outputs[3], outputs[1])
            loss = adv_error * self.adv_weight + \
                con_error * self.con_weight + \
                enc_error * self.enc_weight
            self.loss_info['adv_loss'] = adv_error.item()
            self.loss_info['con_loss'] = con_error.item()
            self.loss_info['enc_loss'] = enc_error.item()
        else:
            loss = self.enc_loss(outputs[2], outputs[0])
        return loss
