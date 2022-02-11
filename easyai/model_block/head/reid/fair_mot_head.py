#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.name_manager.block_name import ActivationType
from easyai.model_block.utility.base_block import *
from easyai.model_block.base_block.common.utility_layer import ActivationLayer
from easyai.model_block.utility.block_registry import REGISTERED_MODEL_HEAD


@REGISTERED_MODEL_HEAD.register_module(HeadType.FairMOTHead)
class FairMOTHead(BaseBlock):

    def __init__(self, in_channels, output_list,
                 activation_name=ActivationType.Swish):
        super().__init__(HeadType.FairMOTHead)
        self.layer_blocks = nn.ModuleList()
        for num_output in output_list:
            fc = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=True),
                ActivationLayer(activation_name),
                nn.Conv2d(64, num_output, kernel_size=1, stride=1, padding=0))
            self.layer_blocks.append(fc)

    def forward(self, x):
        result = []
        for block in self.layer_blocks:
            temp = block(x)
            result.append(temp)
        return result
