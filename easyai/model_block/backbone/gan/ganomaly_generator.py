#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType
from easyai.name_manager.backbone_name import GanBaseModelName
from easyai.model_block.base_block.gan.dc_encoder import DCEncoder
from easyai.model_block.base_block.gan.dc_decoder import DCDecoder
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.backbone_registry import REGISTERED_GAN_G_BACKBONE


@REGISTERED_GAN_G_BACKBONE.register_module(GanBaseModelName.GANomalyGenerator)
class GANomalyGenerator(BaseBackbone):

    def __init__(self, data_channel=3, input_size=32, final_out_channel=100):
        super().__init__(data_channel)
        self.set_name(GanBaseModelName.GANomalyGenerator)
        self.input_size = input_size
        self.final_out_channel = final_out_channel
        self.activation_name = ActivationType.LeakyReLU
        self.first_output = 64

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        encoder1 = DCEncoder(self.input_size, self.data_channel,
                             self.first_output, self.final_out_channel)
        self.add_block_list(encoder1.get_name(), encoder1, self.final_out_channel)

        decoder = DCDecoder(self.input_size, self.final_out_channel,
                            self.first_output, self.data_channel)
        self.add_block_list(decoder.get_name(), decoder, self.data_channel)

        encoder2 = DCEncoder(self.input_size, self.data_channel,
                             self.first_output, self.final_out_channel)
        self.add_block_list(encoder2.get_name(), encoder2, self.final_out_channel)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            # print(key, x.shape)
        return output_list
