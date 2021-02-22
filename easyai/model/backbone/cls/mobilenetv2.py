#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.residual_block import InvertedResidual
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE


__all__ = ['MobileNetV2V10', 'MobileNetV2V10Dilated8']


class MobileNetV2(BaseBackbone):

    def __init__(self, data_channel=3, input_stride=2,
                 num_blocks=(1, 2, 3, 4, 3, 3, 1),
                 out_channels=(16, 24, 32, 64, 96, 160, 320), strides=(1, 2, 2, 2, 1, 2, 1),
                 dilations=(1, 1, 1, 1, 1, 1, 1), expand_ratios=(1, 6, 6, 6, 6, 6, 6),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU6):
        super().__init__(data_channel)
        self.set_name(BackboneName.MobileNetV2_1_0)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activation_name
        self.bnName = bn_name
        self.expand_ratios = expand_ratios
        self.input_stride = input_stride
        self.first_output = 32

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=self.input_stride,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_mobile_layer(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.bnName, self.activationName,
                                   self.expand_ratios[index])
            self.in_channels = self.block_out_channels[-1]

    def make_mobile_layer(self, out_channels, num_blocks, stride, dilation,
                          bnName, activationName, expand_ratio):
        if dilation > 1:
            stride = 1
        down_layers = InvertedResidual(self.in_channels, out_channels, stride=stride,
                                       expand_ratio=expand_ratio, dilation=dilation,
                                       bnName=bnName, activationName=activationName)
        name = "down_%s" % down_layers.get_name()
        self.add_block_list(name, down_layers, out_channels)
        for _ in range(num_blocks - 1):
            layer = InvertedResidual(out_channels, out_channels, stride=1, expand_ratio=expand_ratio,
                                     dilation=dilation, bnName=bnName, activationName=activationName)
            self.add_block_list(layer.get_name(), layer, out_channels)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2_1_0)
class MobileNetV2V10(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=(1, 2, 3, 4, 3, 3, 1))
        self.set_name(BackboneName.MobileNetV2_1_0)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2V10Dilated8)
class MobileNetV2V10Dilated8(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=(1, 2, 3, 4, 3, 3, 1),
                         dilations=(1, 1, 1, 2, 2, 4, 4))
        self.set_name(BackboneName.MobileNetV2V10Dilated8)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV2Down4)
class MobileNetV2Down4(MobileNetV2):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         input_stride=1,
                         num_blocks=(1, 2, 3, 4, 3, 3, 1),
                         strides=(1, 2, 1, 2, 1, 1, 1))
        self.set_name(BackboneName.MobileNetV2Down4)

