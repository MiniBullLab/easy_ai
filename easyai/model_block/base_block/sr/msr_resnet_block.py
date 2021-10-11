#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType
from easyai.model_block.base_block.common.utility_block import ConvActivationBlock
from easyai.model_block.utility.base_block import *


class MSRResNetBlockName():

    ResidualBlockNoBN = "ResidualBlockNoBN"


class ResidualBlockNoBN(BaseBlock):

    def __init__(self, in_channel=64, activation_name=ActivationType.ReLU):
        super().__init__(MSRResNetBlockName.ResidualBlockNoBN)
        self.conv1 = ConvActivationBlock(in_channels=in_channel,
                                         out_channels=in_channel,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         bias=True,
                                         activationName=activation_name)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return identity + out
