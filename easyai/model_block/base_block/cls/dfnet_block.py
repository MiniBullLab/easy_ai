#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.model_block.base_block.common.utility_layer import ActivationLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.utility.base_block import *


class DFNetBlockName():

    BasicBlock = "BasicBlock"


class BasicBlock(BaseBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(DFNetBlockName.BasicBlock)
        self.conv1 = ConvBNActivationBlock(in_channels=inplanes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = ConvBNActivationBlock(in_channels=planes,
                                           out_channels=planes,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False,
                                           bnName=bn_name,
                                           activationName=ActivationType.Linear)
        self.relu = ActivationLayer(activation_name=activation_name)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
