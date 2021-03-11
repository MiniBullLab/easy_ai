#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType
from easyai.base_name.block_name import HeadType
from easyai.model.model_block.base_block.utility.upsample_layer import Upsample
from easyai.model.model_block.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.model_block.base_block.utility.pooling_layer import MultiROIPooling
from easyai.model.model_block.base_block.utility.base_block import *


class MultiROIKeypointHead(BaseBlock):

    def __init__(self, in_channels, keypoint_number,
                 pool_resolution, pool_scales, pool_sampling_ratio,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.MultiROIKeypointHead)
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
                                             activationName=activation_name)
            self.blocks.append((layer_name, temp_block))
            next_feature = layer_features
        self.conv_blocks = nn.Sequential(*blocks)

        self.keypoints_score = torch.nn.ConvTranspose2d(in_channels=next_feature,
                                                        out_channels=keypoint_number,
                                                        kernel_size=4,
                                                        stride=2,
                                                        padding=4//2 - 1)
        self.up = Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.conv_blocks(x)
        x = self.keypoints_score(x)
        x = self.up(x)
        return x
