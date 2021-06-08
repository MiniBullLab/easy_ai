#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.name_manager.block_name import BlockType, LayerType
from easyai.model_block.base_block.common.utility_layer import EmptyLayer
from easyai.model_block.base_block.common.activation_function import ActivationFunction
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.utility_block import BNActivationConvBlock
from easyai.model_block.base_block.common.attention_block import SEConvBlock, SEBlock
from easyai.model_block.utility.base_block import *


class ResidualBlock(BaseBlock):

    def __init__(self, flag, in_channels, out_channels,
                 stride=1, dilation=1, expansion=1, use_short=False,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.ResidualBlock)
        self.residual = nn.Sequential()
        if flag == 0:
            self.residual = nn.Sequential(
                ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=3,
                                      stride=1,
                                      padding=dilation,
                                      dilation=dilation,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear)
            )
        elif flag == 1:
            self.residual = nn.Sequential(
                ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                ConvBNActivationBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=ActivationType.Linear)
            )

        self.shortcut = nn.Sequential()
        if use_short or stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = ConvBNActivationBlock(in_channels=in_channels,
                                                  out_channels=out_channels * expansion,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bnName=bn_name,
                                                  activationName=ActivationType.Linear)
        self.activation = ActivationFunction.get_function(activation_name)

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        out = self.activation(out)
        return out


class ResidualV2Block(BaseBlock):

    def __init__(self, flag, in_channels, out_channels, stride=1, dilation=1, expansion=1,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.ResidualV2Block)
        self.residual = nn.Sequential()
        if flag == 0:
            self.residual = nn.Sequential(
                BNActivationConvBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                BNActivationConvBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=3,
                                      stride=1,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name)
            )
        elif flag == 1:
            self.residual = nn.Sequential(
                BNActivationConvBlock(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                BNActivationConvBlock(in_channels=out_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=dilation,
                                      dilation=dilation,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name),
                BNActivationConvBlock(in_channels=out_channels,
                                      out_channels=out_channels * expansion,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=bn_name,
                                      activationName=activation_name)
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = BNActivationConvBlock(in_channels=in_channels,
                                                  out_channels=out_channels * expansion,
                                                  kernel_size=1,
                                                  stride=stride,
                                                  bias=False,
                                                  bnName=bn_name,
                                                  activationName=activation_name)

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        out = residual + shortcut
        return out


class InvertedResidual(BaseBlock):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, dilation=1,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU6):
        super().__init__(BlockType.InvertedResidual)
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        inter_channels = int(round(in_channels * expand_ratio))
        layers = OrderedDict()
        if expand_ratio != 1:
            # pw
            convBNReLU1 = ConvBNActivationBlock(in_channels=in_channels,
                                                out_channels=inter_channels,
                                                kernel_size=1,
                                                bnName=bnName,
                                                activationName=activationName)
            layer_name = "%s_1" % BlockType.ConvBNActivationBlock
            layers[layer_name] = convBNReLU1
        # dw
        convBNReLU2 = ConvBNActivationBlock(in_channels=inter_channels,
                                            out_channels=inter_channels,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=dilation,
                                            dilation=dilation,
                                            groups=inter_channels,
                                            bnName=bnName,
                                            activationName=activationName)
        # pw-linear
        convBNReLU3 = ConvBNActivationBlock(in_channels=inter_channels,
                                            out_channels=out_channels,
                                            kernel_size=1,
                                            bnName=bnName,
                                            activationName=ActivationType.Linear)

        layer_name = "%s_2" % BlockType.ConvBNActivationBlock
        layers[layer_name] = convBNReLU2
        layer_name = "%s_3" % BlockType.ConvBNActivationBlock
        layers[layer_name] = convBNReLU3
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class InvertedResidualV2(BaseBlock):
    def __init__(self, flag, in_channels, hidden_dim, out_channels,
                 kernel_size, stride, se_type,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.HardSwish):
        super().__init__(BlockType.InvertedResidualV2)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        layers = OrderedDict()
        if flag == 0:
            layer_name = "%s_1" % BlockType.ConvBNActivationBlock
            layers[layer_name] = ConvBNActivationBlock(in_channels=hidden_dim,
                                                       out_channels=hidden_dim,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=(kernel_size - 1) // 2,
                                                       groups=hidden_dim,
                                                       bias=False,
                                                       bnName=bn_name,
                                                       activationName=activation_name)
            if se_type == 0:
                layer_name = "%s_2" % LayerType.EmptyLayer
                layers[layer_name] = EmptyLayer()
            elif se_type == 1:
                layer_name = "%s_2" % BlockType.SEBlock
                layers[layer_name] = SEBlock(hidden_dim, reduction=4,
                                             activate_name2=ActivationType.HardSigmoid)
            elif se_type == 2:
                layer_name = "%s_2" % BlockType.SEConvBlock
                layers[layer_name] = SEConvBlock(hidden_dim, hidden_dim, reduction=4,
                                                 activate_name2=ActivationType.HardSigmoid)
            layer_name = "%s_3" % BlockType.ConvBNActivationBlock
            layers[layer_name] = ConvBNActivationBlock(in_channels=hidden_dim,
                                                       out_channels=out_channels,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False,
                                                       bnName=bn_name,
                                                       activationName=ActivationType.Linear)
        else:
            expand_name = "%s_1" % BlockType.ConvBNActivationBlock
            layers[expand_name] = ConvBNActivationBlock(in_channels=in_channels,
                                                        out_channels=hidden_dim,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        bias=False,
                                                        bnName=bn_name,
                                                        activationName=activation_name)
            bottleneck_name = "%s_2" % BlockType.ConvBNActivationBlock
            layers[bottleneck_name] = ConvBNActivationBlock(in_channels=hidden_dim,
                                                            out_channels=hidden_dim,
                                                            kernel_size=kernel_size,
                                                            stride=stride,
                                                            padding=(kernel_size - 1) // 2,
                                                            groups=hidden_dim,
                                                            bias=False,
                                                            bnName=bn_name,
                                                            activationName=activation_name)
            if se_type == 0:
                layer_name = "%s_3" % LayerType.EmptyLayer
                layers[layer_name] = EmptyLayer()
            elif se_type == 1:
                layer_name = "%s_3" % BlockType.SEBlock
                layers[layer_name] = SEBlock(hidden_dim, reduction=4,
                                             activate_name2=ActivationType.HardSigmoid)
            elif se_type == 2:
                layer_name = "%s_3" % BlockType.SEConvBlock
                layers[layer_name] = SEConvBlock(hidden_dim, hidden_dim, reduction=4,
                                                 activate_name2=ActivationType.HardSigmoid)
            layer_name = "%s_4" % BlockType.ConvBNActivationBlock
            layers[layer_name] = ConvBNActivationBlock(in_channels=hidden_dim,
                                                       out_channels=out_channels,
                                                       kernel_size=1,
                                                       stride=1,
                                                       padding=0,
                                                       bias=False,
                                                       bnName=bn_name,
                                                       activationName=ActivationType.Linear)
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)
