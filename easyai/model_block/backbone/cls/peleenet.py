#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
title = {Pelee: A Real-Time Object Detection System on Mobile Devices},
author = {Wang, Robert J and Li, Xiang and Ling, Charles X},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {1967--1976},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7466-pelee
"""

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.cls.peleenet_block import StemBlock, DenseBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.backbone_registry import REGISTERED_CLS_BACKBONE


__all__ = ['PeleeNet12', 'PeleeNet24', 'PeleeNet16']


class PeleeNet(BaseBackbone):

    def __init__(self, data_channel=3, num_init_features=12,
                 growth_rates=(12, 12, 12, 12), num_blocks=(3, 4, 6, 3),
                 bottleneck_widths=(1, 2, 4, 4),
                 use_transition=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.PeleeNet12)
        assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
        assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
        self.num_init_features = num_init_features
        self.growth_rates = growth_rates
        self.num_blocks = num_blocks
        self.bottleneck_widths = bottleneck_widths
        self.use_transition = use_transition
        self.activation_name = activation_name
        self.bn_name = bn_name

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        stem = StemBlock(self.data_channel, self.num_init_features,
                         bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(stem.get_name(), stem, self.num_init_features)

        self.in_channels = self.num_init_features
        for index, num_block in enumerate(self.num_blocks):
            self.make_peleenet_layer(num_block, self.growth_rates[index],
                                     self.bottleneck_widths[index],
                                     bn_name=self.bn_name,
                                     activation=self.activation_name)
            self.in_channels = self.block_out_channels[-1]
            if self.use_transition:
                temp_conv = ConvBNActivationBlock(in_channels=self.in_channels,
                                                  out_channels=self.in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=False,
                                                  bnName=self.bn_name,
                                                  activationName=self.activation_name)
                self.add_block_list(temp_conv.get_name(), temp_conv, self.in_channels)
            if index != len(self.num_blocks) - 1:
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                self.add_block_list(LayerType.GlobalAvgPool, avg_pool, self.block_out_channels[-1])

    def make_peleenet_layer(self, num_block,
                            growth_rate, bottleneck_width,
                            bn_name, activation):
        for i in range(num_block):
            temp_input_channel = self.in_channels + i * growth_rate
            layer = DenseBlock(temp_input_channel, growth_rate, bottleneck_width,
                               bn_name=bn_name, activation_name=activation)
            temp_output_channel = temp_input_channel + growth_rate
            self.add_block_list(layer.get_name(), layer, temp_output_channel)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.PeleeNet12)
class PeleeNet12(PeleeNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=12,
                         growth_rates=(12, 12, 12, 12),
                         num_blocks=(3, 4, 6, 3),
                         bottleneck_widths=(1, 2, 4, 4))
        self.set_name(BackboneName.PeleeNet12)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.PeleeNet24)
class PeleeNet24(PeleeNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=24,
                         growth_rates=(24, 24, 24, 24),
                         num_blocks=(3, 4, 6, 3),
                         bottleneck_widths=(1, 2, 4, 4))
        self.set_name(BackboneName.PeleeNet24)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.PeleeNet16)
class PeleeNet16(PeleeNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=16,
                         growth_rates=(16, 16, 16, 16),
                         num_blocks=(3, 4, 8, 6),
                         bottleneck_widths=(1, 2, 4, 4))
        self.set_name(BackboneName.PeleeNet16)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.PeleeNetTransition24)
class PeleeNetTransition24(PeleeNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_init_features=24,
                         growth_rates=(24, 24, 24, 24),
                         num_blocks=(3, 4, 6, 3),
                         bottleneck_widths=(1, 2, 4, 4),
                         use_transition=True)
        self.set_name(BackboneName.PeleeNetTransition24)
