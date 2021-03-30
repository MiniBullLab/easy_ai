#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType
from easyai.base_name.backbone_name import GanBaseModelName
from easyai.model.model_block.base_block.gan.dc_encoder import DCEncoder
from easyai.model.model_block.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.model_block.backbone.utility.base_backbone import *
from easyai.model.model_block.backbone.utility.backbone_registry import REGISTERED_GAN_D_BACKBONE


@REGISTERED_GAN_D_BACKBONE.register_module(GanBaseModelName.GANomalyDiscriminator)
class GANomalyDiscriminator(BaseBackbone):

    def __init__(self, data_channel=3, input_size=32):
        super().__init__(data_channel)
        self.set_name(GanBaseModelName.GANomalyDiscriminator)
        self.input_size = input_size
        self.activation_name = ActivationType.ReLU
        self.first_output = 64

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        encoder = DCEncoder(self.input_size, self.data_channel,
                            self.first_output, 1)
        layers = list(encoder.block.children())
        features = nn.Sequential(*layers[:-1])
        self.add_block_list(encoder.get_name(), features, encoder.get_output_channel())

        conv = ConvActivationBlock(in_channels=encoder.get_output_channel(),
                                   out_channels=1,
                                   kernel_size=4,
                                   stride=1,
                                   padding=0,
                                   bias=False,
                                   activationName=ActivationType.Sigmoid)
        self.add_block_list(conv.get_name(), conv, 1)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            # print(key, x.shape)
        return output_list
