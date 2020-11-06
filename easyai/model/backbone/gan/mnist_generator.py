#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType
from easyai.base_name.backbone_name import GanBaseModelName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import FcActivationBlock
from easyai.model.backbone.utility.registry import REGISTERED_GAN_G_BACKBONE


@REGISTERED_GAN_G_BACKBONE.register_module(GanBaseModelName.MNISTGenerator)
class MNISTGenerator(BaseBackbone):

    def __init__(self, data_channel=3, final_out_channel=1, activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(GanBaseModelName.MNISTGenerator)
        self.final_out_channel = final_out_channel
        self.activation_name = activation_name

        self.first_output = 256
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = FcActivationBlock(self.data_channel, self.first_output,
                                   activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = FcActivationBlock(self.in_channel, self.in_channel,
                                   activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, self.in_channel)

        layer3 = FcActivationBlock(self.in_channel, self.final_out_channel,
                                   activationName=ActivationType.Tanh)

        self.add_block_list(layer3.get_name(), layer3, self.final_out_channel)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
        return output_list

