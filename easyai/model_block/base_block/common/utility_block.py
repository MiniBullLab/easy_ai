#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.model_block.base_block.common.activation_function import ActivationFunction
from easyai.model_block.base_block.common.normalization_layer import NormalizationFunction
from easyai.model_block.base_block.common.pooling_layer import MyAvgPool2d
from easyai.model_block.utility.base_block import *


class ConvBNActivationBlock1d(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bnName=NormalizationType.BatchNormalize1d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock1d)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional1d, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNBlock1d(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bnName=NormalizationType.BatchNormalize1d):
        super().__init__(BlockType.ConvBNBlock1d)
        conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional1d, conv),
            (bnName, bn)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvActivationBlock)
        self.activation_name = activationName
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNActivationBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNActivationBlock)
        self.bn_name = bnName
        self.activation_name = activationName
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x

    # def forward_fuse(self, x):
    #     return self.activation(self.conv(x))


class ConvDropBNActivationBlock(BaseBlock):

    def __init__(self, p, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.ConvDropBNActivationBlock)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        drop = nn.Dropout(p=p)
        bn = NormalizationFunction.get_function(bn_name, out_channels)
        activation = ActivationFunction.get_function(activation_name)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.Convolutional, conv),
            (LayerType.Dropout, drop),
            (bn_name, bn),
            (activation_name, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ConvBNACTWithPoolBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=1, padding=0,
                 dilation=1, groups=1, bias=False,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ConvBNACTWithPoolBlock)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
        self.conv = ConvBNActivationBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias,
                                          bnName=bnName,
                                          activationName=activationName)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class BNActivationConvBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.BNActivationConvBlock)
        bn = NormalizationFunction.get_function(bnName, in_channels)
        activation = ActivationFunction.get_function(activationName)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        self.block = nn.Sequential(OrderedDict([
            (bnName, bn),
            (activationName, activation),
            (LayerType.Convolutional, conv)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class ActivationConvBNBlock(BaseBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.ActivationConvBNBlock)
        activation = ActivationFunction.get_function(activationName)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_channels)
        self.block = nn.Sequential(OrderedDict([
            (activationName, activation),
            (LayerType.Convolutional, conv),
            (bnName, bn)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class FcBNActivationBlock(BaseBlock):

    def __init__(self, in_features, out_features, bias=False,
                 bnName=NormalizationType.BatchNormalize1d,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.FcBNActivationBlock)
        fc = nn.Linear(in_features, out_features, bias=bias)
        bn = NormalizationFunction.get_function(bnName, out_features)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.FcLinear, fc),
            (bnName, bn),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


class FcActivationBlock(BaseBlock):

    def __init__(self, in_features, out_features, bias=True,
                 activationName=ActivationType.ReLU):
        super().__init__(BlockType.FcActivationBlock)
        fc = nn.Linear(in_features, out_features, bias=bias)
        activation = ActivationFunction.get_function(activationName)
        self.block = nn.Sequential(OrderedDict([
            (LayerType.FcLinear, fc),
            (activationName, activation)
        ]))

    def forward(self, x):
        x = self.block(x)
        return x


if __name__ == "__main__":
    pass
