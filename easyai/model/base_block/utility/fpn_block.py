#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.base_block.utility.base_block import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock


class FPNBlock(BaseBlock):

    def __init__(self, layers, channels, out_channels,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.FPNBlock)
        self.layers = [int(x) for x in layers.split(',') if x.strip()]
        self.in_channels_list = [int(x) for x in channels.split(',') if x.strip()]
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for index, in_channel in enumerate(self.in_channels_list):
            temp1 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bnName=bnName,
                                          activationName=activationName)
            temp2 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bnName=bnName,
                                          activationName=activationName)
            self.inner_blocks.append(temp1)
            self.layer_blocks.append(temp2)

    def forward(self, layer_outputs, base_outputs):
        # print(self.layers)
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = torch.cat(temp_layer_outputs, 1)
        return x
