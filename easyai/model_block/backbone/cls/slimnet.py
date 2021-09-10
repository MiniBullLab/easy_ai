#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.separable_conv_block import DepthwiseConv2dBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SlimNet)
class SlimNet(BaseBackbone):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.SlimNet)
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.first_output = 16

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = DepthwiseConv2dBlock(in_channel=self.first_output,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, self.first_output)

        layer3 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=32,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, 32)

        layer4 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(layer4.get_name(), layer4, self.block_out_channels[-1])

        layer5 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=32,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer5.get_name(), layer5, 32)

        layer6 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(layer6.get_name(), layer6, self.block_out_channels[-1])

        layer7 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=64,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer7.get_name(), layer7, 64)

        layer8 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bn_name=self.bn_name,
                                      activation_name=self.activation_name)
        self.add_block_list(layer8.get_name(), layer8, self.block_out_channels[-1])

        layer9 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=64,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer9.get_name(), layer9, 64)

        layer10 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bn_name=self.bn_name,
                                       activation_name=self.activation_name)
        self.add_block_list(layer10.get_name(), layer10, self.block_out_channels[-1])

        layer11 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer11.get_name(), layer11, 128)

        layer12 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bn_name=self.bn_name,
                                       activation_name=self.activation_name)
        self.add_block_list(layer12.get_name(), layer12, self.block_out_channels[-1])

        layer13 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=128,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer13.get_name(), layer13, 128)

        layer14 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bn_name=self.bn_name,
                                       activation_name=self.activation_name)
        self.add_block_list(layer14.get_name(), layer14, self.block_out_channels[-1])

        layer15 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer15.get_name(), layer15, 256)

        layer16 = DepthwiseConv2dBlock(in_channel=self.block_out_channels[-1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bn_name=self.bn_name,
                                       activation_name=self.activation_name)
        self.add_block_list(layer16.get_name(), layer16, self.block_out_channels[-1])

        layer17 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                        out_channels=256,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(layer17.get_name(), layer17, 256)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
            print(x.shape)
        return output_list
