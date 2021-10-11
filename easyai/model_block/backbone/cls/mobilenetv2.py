#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.residual_block import InvertedResidual
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


__all__ = ['MobileNetV2V01', 'MobileNetV2V05', 'MobileNetV2V10',
           'MobileNetV2V10Dilated8', 'MobileNetV2Down4']


class MobileNetV2(BaseBackbone):

    def __init__(self, data_channel=3, input_stride=2, width_mult=1.,
                 num_blocks=(1, 2, 3, 4, 3, 3, 1),
                 out_channels=(16, 24, 32, 64, 96, 160, 320),
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 dilations=(1, 1, 1, 1, 1, 1, 1),
                 expand_ratios=(1, 6, 6, 6, 6, 6, 6),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU6):
        super().__init__(data_channel)
        self.set_name(BackboneName.MobileNetV2_1_0)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.expand_ratios = expand_ratios
        self.input_stride = input_stride
        self.width_mult = width_mult
        self.first_output = self.make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=self.input_stride,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            output_channel = self.make_divisible(self.out_channels[index] * self.width_mult, 4 if self.width_mult == 0.1 else 8)
            self.make_mobile_layer(output_channel, self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bn_name, self.activation_name,
                                   self.expand_ratios[index])
            self.in_channels = self.block_out_channels[-1]

    def make_mobile_layer(self, out_channels, num_blocks, stride, dilation,
                          bn_name, activation_name, expand_ratio):
        if dilation > 1:
            stride = 1
        down_layers = InvertedResidual(self.in_channels, out_channels, stride=stride,
                                       expand_ratio=expand_ratio, dilation=dilation,
                                       bnName=bn_name, activationName=activation_name)
        name = "down_%s" % down_layers.get_name()
        self.add_block_list(name, down_layers, out_channels)
        for _ in range(num_blocks - 1):
            layer = InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=expand_ratio,
                                     dilation=dilation, bnName=bn_name, activationName=activation_name)
            self.add_block_list(layer.get_name(), layer, out_channels)

    def make_divisible(self, v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            # print(key, x.shape)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2_0_1)
class MobileNetV2V01(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         width_mult=0.1)
        self.set_name(BackboneName.MobileNetV2_0_1)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2_0_5)
class MobileNetV2V05(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         width_mult=0.5)
        self.set_name(BackboneName.MobileNetV2_0_5)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2_1_0)
class MobileNetV2V10(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel)
        self.set_name(BackboneName.MobileNetV2_1_0)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2V10Dilated8)
class MobileNetV2V10Dilated8(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         dilations=(1, 1, 1, 2, 2, 4, 4))
        self.set_name(BackboneName.MobileNetV2V10Dilated8)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2Down4)
class MobileNetV2Down4(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         input_stride=1,
                         strides=(1, 2, 1, 2, 1, 1, 1))
        self.set_name(BackboneName.MobileNetV2Down4)

