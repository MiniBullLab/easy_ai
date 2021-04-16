#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.model.model_block.base_block.utility.residual_block import ResidualV2Block


class HourGlassBlock(BaseBlock):

    def __init__(self, depth=4, feature_channel=96,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.HourGlassBlock)
        self.depth = depth
        self.feature_channel = feature_channel
        self.bn_name = bn_name
        self.activation_name = activation_name
        self._generate_network(self.depth, self.feature_channel)

    def _generate_network(self, level, num_features):
        self.add_module('b1_' + str(level),
                        ResidualV2Block(1, self.feature_channel, self.feature_channel // 2,
                                        stride=1, expansion=2,
                                        bn_name=self.bn_name,
                                        activation_name=self.activation_name))

        self.add_module('b2_' + str(level),
                        ResidualV2Block(1, self.feature_channel, self.feature_channel // 2,
                                        stride=1, expansion=2,
                                        bn_name=self.bn_name,
                                        activation_name=self.activation_name))

        if level > 1:
            self._generate_network(level - 1, num_features)
        else:
            self.add_module('b2_plus_' + str(level),
                            ResidualV2Block(1, self.feature_channel, self.feature_channel // 2,
                                            stride=1, expansion=2,
                                            bn_name=self.bn_name,
                                            activation_name=self.activation_name))

        self.add_module('b3_' + str(level),
                        ResidualV2Block(1, self.feature_channel, self.feature_channel // 2,
                                        stride=1, expansion=2,
                                        bn_name=self.bn_name,
                                        activation_name=self.activation_name))

        self.add_module('avg_pool_' + str(level), nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        self.add_module('up2_' + str(level), nn.ConvTranspose2d(in_channels=num_features, out_channels=num_features,
                                                                kernel_size=2, stride=2, padding=0, groups=num_features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        # low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['avg_pool_' + str(level)](inp)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest')
        up2 = self._modules['up2_' + str(level)](low3)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)