#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model_block.utility.base_block import *
from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.upsample_layer import DeConvBNActivationBlock

from easy_pc.name_manager.pc_block_name import PCNeckType


class SecondFPNNeck(BaseBlock):

    def __init__(self, down_layers, down_channels,
                 out_channels, upsample_strides=(1, 2, 4),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(PCNeckType.SecondFPNNeck)
        # print(self.down_layers)
        self.down_layers = [int(x) for x in down_layers.split(',') if x.strip()]
        assert len(self.down_layers) == len(down_channels)
        assert len(self.upsample_strides) == len(down_channels)
        self.layer_blocks = nn.ModuleList()

        for index, in_channel in enumerate(self.down_channels):
            if self.upsample_strides[index] > 1:
                temp = DeConvBNActivationBlock(in_channels=in_channel,
                                               out_channels=out_channels,
                                               kernel_size=upsample_strides[index],
                                               stride=upsample_strides[index],
                                               padding=0,
                                               bn_name=bn_name,
                                               activation_name=activation_name)
            else:
                temp = ConvBNActivationBlock(in_channels=in_channel,
                                             out_channels=out_channels,
                                             kernel_size=upsample_strides[index],
                                             stride=1,
                                             padding=0,
                                             bnName=bn_name,
                                             activationName=activation_name)
            self.layer_blocks.append(temp)

    def forward(self, layer_outputs, base_outputs):
        results = []
        input_features = [layer_outputs[i] if i < 0 else base_outputs[i]
                          for i in self.down_layers]
        for feature, layer_block in zip(input_features, self.layer_blocks):
            temp_result = layer_block(feature)
            results.append(temp_result)
            # print(results[-1].shape)
        return torch.cat(results, dim=1)

