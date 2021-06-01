#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.model_block.base_block.common.utility_layer import ActivationLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.attention_block import SEBlock
from easyai.model_block.utility.base_block import *


class SeNetBlockName():

    BasicResidualSEBlock = "basicResidualSEBlock"
    BottleneckResidualSEBlock = "bottleneckResidualSEBlock"


class BasicResidualSEBlock(BaseBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, reduction=16,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(SeNetBlockName.BasicResidualSEBlock)

        self.residual = nn.Sequential(
            ConvBNActivationBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=out_channels,
                                  out_channels=out_channels * self.expansion,
                                  kernel_size=3,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                                  out_channels=out_channels * self.expansion,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)
        self.se_block = SEBlock(out_channels * self.expansion, reduction)

        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        excitation = self.se_block(residual)
        x = residual * excitation.expand_as(residual) + shortcut
        x = self.relu(x)
        return x


class BottleneckResidualSEBlock(BaseBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, reduction=16,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(SeNetBlockName.BottleneckResidualSEBlock)

        self.residual = nn.Sequential(
            ConvBNActivationBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1,
                                  bnName=bn_name,
                                  activationName=activation_name),
            ConvBNActivationBlock(in_channels=out_channels,
                                  out_channels=out_channels * self.expansion,
                                  kernel_size=1,
                                  bnName=bn_name,
                                  activationName=activation_name)
        )

        self.se_block = SEBlock(out_channels * self.expansion, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                                  out_channels=out_channels * self.expansion,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)
        self.relu = ActivationLayer(activation_name=activation_name, inplace=False)

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        excitation = self.se_block(residual)
        x = residual * excitation.expand_as(residual) + shortcut
        x = self.relu(x)
        return x
