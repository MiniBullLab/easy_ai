#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType
from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.common.utility_block import ConvActivationBlock
from easyai.model_block.base_block.common.pooling_layer import MultiROIPooling
from easyai.model_block.utility.base_block import *


class MultiROIMaskHead(BaseBlock):

    def __init__(self, in_channels,
                 pool_resolution, pool_scales, pool_sampling_ratio,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.MultiROIMaskHead)
        self.pooling = MultiROIPooling(out_channels=in_channels,
                                       output_size=(pool_resolution, pool_resolution),
                                       scales=pool_scales,
                                       sampling_ratio=pool_sampling_ratio)

        conv_layers = (256, 256, 256, 256)
        next_feature = in_channels
        blocks = []
        for layer_idx, layer_features in enumerate(conv_layers, 1):
            layer_name = "conv_fcn{}".format(layer_idx)
            temp_block = ConvActivationBlock(next_feature, layer_features,
                                             kernel_size=3, stride=1, padding=1,
                                             dilation=1,
                                             activationName=activation_name)
            self.blocks.append((layer_name, temp_block))
            next_feature = layer_features
        self.conv_blocks = nn.Sequential(*blocks)

        self.conv_mask = nn.ConvTranspose2d(in_channels=next_feature,
                                            out_channels=conv_layers[-1],
                                            kernel_size=2,
                                            stride=2,
                                            padding=0)
        self.mask_fcn_logits = nn.Conv2d(conv_layers[-1], 2, 1, 1, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.conv_blocks(x)
        x = self.conv_mask(x)
        x = self.mask_fcn_logits(x)
        return x
