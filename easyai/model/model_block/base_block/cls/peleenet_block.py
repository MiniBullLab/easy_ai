#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.model_block.base_block.utility.pooling_layer import MyMaxPool2d


class PeleeNetBlockName():

    StemBlock = "stemBlock"
    DenseBlock = "denseBlock"


class StemBlock(BaseBlock):

    def __init__(self, data_channel, num_init_features,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PeleeNetBlockName.StemBlock)

        num_stem_features = int(num_init_features/2)

        self.stem1 = ConvBNActivationBlock(data_channel, num_init_features,
                                           kernel_size=3, stride=2, padding=1, bias=False,
                                           bnName=bn_name, activationName=activation_name)
        self.stem2a = ConvBNActivationBlock(num_init_features, num_stem_features,
                                            kernel_size=1, stride=1, padding=0, bias=False,
                                            bnName=bn_name, activationName=activation_name)
        self.stem2b = ConvBNActivationBlock(num_stem_features, num_init_features,
                                            kernel_size=3, stride=2, padding=1, bias=False,
                                            bnName=bn_name, activationName=activation_name)
        self.stem3 = ConvBNActivationBlock(2*num_init_features, num_init_features,
                                           kernel_size=1, stride=1, padding=0,
                                           bnName=bn_name, activationName=activation_name)
        self.pool = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class DenseBlock(BaseBlock):

    def __init__(self, num_input_features, growth_rate, bottleneck_width,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PeleeNetBlockName.DenseBlock)

        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ',inter_channel)

        self.branch1a = ConvBNActivationBlock(num_input_features, inter_channel,
                                              kernel_size=1, bias=False,
                                              bnName=bn_name, activationName=activation_name)
        self.branch1b = ConvBNActivationBlock(inter_channel, growth_rate,
                                              kernel_size=3, padding=1, bias=False,
                                              bnName=bn_name, activationName=activation_name)

        self.branch2a = ConvBNActivationBlock(num_input_features, inter_channel,
                                              kernel_size=1, bias=False,
                                              bnName=bn_name, activationName=activation_name)
        self.branch2b = ConvBNActivationBlock(inter_channel, growth_rate,
                                              kernel_size=3, padding=1, bias=False,
                                              bnName=bn_name, activationName=activation_name)
        self.branch2c = ConvBNActivationBlock(growth_rate, growth_rate,
                                              kernel_size=3, padding=1, bias=False,
                                              bnName=bn_name, activationName=activation_name)

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)

        return torch.cat([x, branch1, branch2], 1)
