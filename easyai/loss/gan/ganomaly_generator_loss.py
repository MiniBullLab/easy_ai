#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.common.common_loss import l2_loss
from easyai.torch_utility.torch_vision.torchvision_visualizer import TorchVisionVisualizer
from easyai.config.utility.image_task_config import ImageTaskConfig
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

        self.config = ImageTaskConfig("gan")
        self.vision = TorchVisionVisualizer()
        self.save_dir = os.path.join(self.config.root_save_dir, "generate")
        self.save_index = 0
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.max_socre = 0
        self.loss_info = {'adv_loss': 0, 'con_loss': 0, 'enc_loss': 0}

    def forward(self, outputs, targets=None):
        if targets is not None:
            if len(outputs) == 3:
                loss = self.enc_loss(outputs[2], outputs[0])
                if loss.item() > self.max_socre:
                    self.max_socre = loss.item()
                    print("max_score:", self.max_socre)
            else:
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
            save_name = "gen_%d.png" % self.save_index
            save_path = os.path.join(self.save_dir, save_name)
            self.vision.save_current_images(outputs[1], save_path)
            self.save_index = (self.save_index + 1) % 500
            loss = torch.mean(torch.pow((outputs[2] - outputs[0]), 2), dim=1)
        return loss
