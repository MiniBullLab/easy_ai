#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.model_block.base_block.utility.activation_function import ActivationFunction
from easyai.model.model_block.base_block.utility.normalization_layer import NormalizationFunction
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.model_block.base_block.utility.base_block import *


class Upsample(BaseBlock):

    def __init__(self, scale_factor=1.0, mode='bilinear', align_corners=False):
        super().__init__(LayerType.Upsample)
        self.scale_factor = scale_factor
        self.mode = mode
        self.image_size = (416, 416)
        self.gain = 1/32
        self.is_onnx_export = False
        if mode in ('nearest', 'area'):
            self.align_corners = None
        else:
            self.align_corners = align_corners
        if self.is_onnx_export:  # explicitly state size, avoid scale_factor
            self.layer = nn.Upsample(size=tuple(int(x * self.gain) for x in self.image_size))
        else:
            self.layer = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode,
                                     align_corners=self.align_corners)

    def forward(self, x):
        x = self.layer(x)
        return x


class DenseUpsamplingConvBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, upscale_factor=2,
                 bn_name=NormalizationType.BatchNormalize1d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.DenseUpsamplingConvBlock)
        self.conv = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False,
                                          bnName=bn_name,
                                          activationName=activation_name)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.output_channel = out_channels // (upscale_factor ** 2)

    def get_output_channel(self):
        return self.output_channel

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class DeConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, dilation=1, bias=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.DeConvBNActivationBlock)
        deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding,
                                    groups=groups,
                                    dilation=dilation,
                                    bias=bias)
        bn = NormalizationFunction.get_function(bn_name, out_channels)
        activation = ActivationFunction.get_function(activation_name)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.ConvTranspose, deconv),
            (bn_name, bn),
            (activation_name, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x