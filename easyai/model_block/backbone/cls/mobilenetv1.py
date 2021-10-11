#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.separable_conv_block import SeparableConv2dBNActivation
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


__all__ = ['MobileNetV1']


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV1)
class MobileNetV1(BaseBackbone):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU6):
        super().__init__(data_channel)
        self.set_name(BackboneName.MobileNetV1)
        self.activation_name = activation_name
        self.bn_name = bn_name

        self.first_output = 32

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bnName,
                                       activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = SeparableConv2dBNActivation(inplanes=self.first_output,
                                             planes=64,
                                             kernel_size=3,
                                             stride=1,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, 64)

        layer3 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=128,
                                             kernel_size=3,
                                             stride=2,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, 128)

        layer4 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=128,
                                             kernel_size=3,
                                             stride=2,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer4.get_name(), layer4, 128)

        layer5 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=128,
                                             kernel_size=3,
                                             stride=1,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer5.get_name(), layer5, 128)

        layer6 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=256,
                                             kernel_size=3,
                                             stride=2,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer6.get_name(), layer6, 256)

        layer7 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=256,
                                             kernel_size=3,
                                             stride=1,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer7.get_name(), layer7, 256)

        layer8 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=512,
                                             kernel_size=3,
                                             stride=2,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer8.get_name(), layer8, 512)
        for _ in range(5):
            temp_layer = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                                     planes=512,
                                                     kernel_size=3,
                                                     stride=1,
                                                     relu_first=False,
                                                     bn_name=self.bn_name,
                                                     activation_name=self.activation_name)
            self.add_block_list(temp_layer.get_name(), temp_layer, 512)

        layer9 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                             planes=1024,
                                             kernel_size=3,
                                             stride=2,
                                             relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(layer9.get_name(), layer9, 1024)

        layer10 = SeparableConv2dBNActivation(inplanes=self.block_out_channels[-1],
                                              planes=1024,
                                              kernel_size=3,
                                              stride=1,
                                              relu_first=False,
                                              bn_name=self.bn_name,
                                              activation_name=self.activation_name)
        self.add_block_list(layer10.get_name(), layer10, 1024)

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            print(key, x.shape)
        return output_list
