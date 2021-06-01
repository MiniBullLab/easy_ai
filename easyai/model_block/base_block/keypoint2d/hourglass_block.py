#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.name_manager.block_name import BlockType
from easyai.model_block.utility.base_block import *
from easyai.model_block.base_block.common.residual_block import ResidualV2Block


class HourglassBlock(BaseBlock):

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

        self.add_module('avg_pool_' + str(level), nn.AvgPool2d(kernel_size=2, stride=2,
                                                               padding=0, ceil_mode=False))

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


# class HourglassBlock(BaseBlock):
#     def __init__(self, depth=4, feature_channel=96, increase=0,
#                  bn_name=NormalizationType.BatchNormalize2d,
#                  activation_name=ActivationType.ReLU):
#         super().__init__(BlockType.HourGlassBlock)
#         nf = feature_channel + increase
#         self.up1 = ResidualV2Block(1, feature_channel,
#                                    feature_channel // 2,
#                                    stride=1, expansion=2,
#                                    bn_name=bn_name,
#                                    activation_name=activation_name)
#         # Lower branch
#         self.pool1 = MyMaxPool2d(2, 2)
#         self.low1 = ResidualV2Block(1, feature_channel, nf // 2,
#                                     stride=1, expansion=2,
#                                     bn_name=bn_name,
#                                     activation_name=activation_name)
#         self.depth = depth
#         # Recursive hourglass
#         if self.depth > 1:
#             self.low2 = HourglassBlock(depth-1, nf)
#         else:
#             self.low2 = ResidualV2Block(1, nf, nf // 2,
#                                         stride=1, expansion=2,
#                                         bn_name=bn_name,
#                                         activation_name=activation_name)
#         self.low3 = ResidualV2Block(1, nf, feature_channel // 2,
#                                     stride=1, expansion=2,
#                                     bn_name=bn_name,
#                                     activation_name=activation_name)
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
#
#     def forward(self, x):
#         up1 = self.up1(x)
#         pool1 = self.pool1(x)
#         low1 = self.low1(pool1)
#         low2 = self.low2(low1)
#         low3 = self.low3(low2)
#         up2 = self.up2(low3)
#         return up1 + up2
