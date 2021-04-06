#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.model_block.base_block.utility.utility_layer import ActivationLayer
from easyai.model.model_block.base_block.utility.pooling_layer import MyMaxPool2d


class FPNBlock(BaseBlock):

    def __init__(self, down_layers, down_channels, out_channels,
                 use_pooling=True, up_mode="nearest",
                 bnName=NormalizationType.EmptyNormalization,
                 activationName=ActivationType.Linear):
        super().__init__(BlockType.FPNBlock)
        # print(self.down_layers)
        self.down_layers = [int(x) for x in down_layers.split(',') if x.strip()]
        self.down_channels = down_channels
        assert len(self.down_layers) == len(self.down_channels)
        self.up_mode = up_mode
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for index, in_channel in enumerate(self.down_channels):
            if in_channel == 0:
                continue
            temp1 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bnName=bnName,
                                          activationName=activationName)
            temp2 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bnName=bnName,
                                          activationName=activationName)
            self.inner_blocks.append(temp1)
            self.layer_blocks.append(temp2)
        self.layer_count = len(self.inner_blocks)
        if use_pooling:
            self.pooling_layer = MyMaxPool2d(kernel_size=1, stride=2)
        else:
            self.pooling_layer = None

    def forward(self, layer_outputs, base_outputs):
        results = []
        input_features = [layer_outputs[i] if i < 0 else base_outputs[i]
                          for i in self.layers]
        last_inner = self.inner_blocks[-1](input_features[-1])
        results.append(self.layer_blocks[-1](last_inner))
        for feature, inner_block, layer_block in zip(
                input_features[:-1][::-1],
                self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode=self.up_mode)
            inner_lateral = inner_block(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, layer_block(last_inner))
        if self.pooling_layer is not None:
            last_results = self.pooling_layer(results[-1])
            results.append(last_results)

        return tuple(results)


class FPNV2Block(BaseBlock):

    def __init__(self, down_layers, down_channels, out_channels,
                 use_pooling=False, up_mode="bilinear",
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.FPNV2Block)
        # print(self.down_layers)
        self.down_layers = [int(x) for x in down_layers.split(',') if x.strip()]
        self.down_channels = down_channels
        assert len(self.down_layers) == len(self.down_channels)
        self.layer_count = len(self.down_layers)
        self.up_mode = up_mode
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for index, in_channel in enumerate(self.down_channels):
            if in_channel == 0:
                continue
            if index == self.layer_count - 1:
                temp1 = ConvBNActivationBlock(in_channels=in_channel,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              bnName=bnName,
                                              activationName=ActivationType.Linear)
            else:
                temp1 = ConvBNActivationBlock(in_channels=in_channel,
                                              out_channels=out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              bnName=bnName,
                                              activationName=activationName)
            temp2 = ConvBNActivationBlock(in_channels=in_channel,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bnName=bnName,
                                          activationName=activationName)
            self.inner_blocks.append(temp1)
            self.layer_blocks.append(temp2)
        self.layer_count = len(self.inner_blocks)
        if use_pooling:
            self.pooling_layer = MyMaxPool2d(kernel_size=1, stride=2)
        else:
            self.pooling_layer = None

        self.activation = ActivationLayer(activation_name=activationName)

    def forward(self, layer_outputs, base_outputs):
        results = []
        input_features = [layer_outputs[i] if i < 0 else base_outputs[i]
                          for i in self.layers]
        last_inner = self.inner_blocks[-1](input_features[-1])
        results.append(self.layer_blocks[-1](last_inner))
        for feature, inner_block, layer_block in zip(
                input_features[:-1][::-1],
                self.inner_blocks[:-1][::-1],
                self.layer_blocks[:-1][::-1]):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode=self.up_mode)
            inner_lateral = inner_block(feature)
            last_inner = inner_lateral + inner_top_down
            last_inner = self.activation(last_inner)
            results.insert(0, layer_block(last_inner))
        if self.pooling_layer is not None:
            last_results = self.pooling_layer(results[-1])
            results.append(last_results)

        return tuple(results)
