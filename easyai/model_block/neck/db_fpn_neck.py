#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager.block_name import ActivationType, NormalizationType
from easyai.config.name_manager.block_name import NeckType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.upsample_layer import Upsample
from easyai.model_block.base_block.common.base_block import *


class DBFPNNeck(BaseBlock):

    def __init__(self, down_layers, down_channels, out_channels, up_mode="nearest",
                 bn_name=NormalizationType.EmptyNormalization,
                 activation_name=ActivationType.Linear):
        super().__init__(NeckType.DBFPNNeck)
        # print(self.down_layers)
        self.down_layers = [int(x) for x in down_layers.split(',') if x.strip()]
        self.down_channels = down_channels
        assert len(self.down_layers) == len(self.down_channels)
        self.up_mode = up_mode
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index, in_channel in enumerate(self.down_channels):
            if in_channel == 0:
                continue
            temp1 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bnName=bn_name,
                                          activationName=activation_name)
            temp2 = ConvBNActivationBlock(in_channels=out_channels,
                                          out_channels=out_channels // 4,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bnName=bn_name,
                                          activationName=activation_name)
            self.inner_blocks.append(temp1)
            self.layer_blocks.append(temp2)
        self.layer_count = len(self.inner_blocks)
        for index in range(self.layer_count - 1, 0, -1):
            temp_up = Upsample(2 ** index, up_mode)
            self.up_blocks.append(temp_up)

    def forward(self, layer_outputs, base_outputs):
        results = []
        input_features = [layer_outputs[i] if i < 0 else base_outputs[i]
                          for i in self.layers]
        last_inner = self.inner_blocks[-1](input_features[-1])
        last_result = self.layer_blocks[-1](last_inner)
        results.append(self.up_blocks[0](last_result))
        for index, feature, inner_block, layer_block in enumerate(zip(
                input_features[:-1][::-1],
                self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]), 1):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode=self.up_mode)
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            temp_result = layer_block(last_inner)
            if index == self.layer_count - 1:
                results.append(temp_result)
            else:
                results.append(self.up_blocks[index](temp_result))
        return torch.cat(results, axis=1)

