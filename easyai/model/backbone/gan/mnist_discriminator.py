#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.base_name.backbone_name import GanBaseModelName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import FcActivationBlock
from easyai.model.backbone.utility.registry import REGISTERED_GAN_D_BACKBONE


@REGISTERED_GAN_D_BACKBONE.register_module(GanBaseModelName.MNISTDiscriminator)
class MNISTDiscriminator(BaseBackbone):

    def __init__(self, data_channel=3, activation_name=ActivationType.LeakyReLU):
        super().__init__(data_channel)
        self.set_name(GanBaseModelName.MNISTDiscriminator)
        self.activation_name = activation_name

        self.first_output = 256
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = FcActivationBlock(self.data_channel, self.first_output,
                                   activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        self.in_channel = self.first_output

        layer2 = FcActivationBlock(self.in_channel, self.in_channel,
                                   activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, self.in_channel)

        layer3 = FcActivationBlock(self.in_channel, 1, activationName=ActivationType.Sigmoid)
        self.add_block_list(layer3.get_name(), layer3, 1)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            # print(key, x.shape)
        return output_list

