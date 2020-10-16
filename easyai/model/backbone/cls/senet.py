#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""senet in pytorch
[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.backbone.utility.base_backbone import *
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.cls.senet_block import BasicResidualSEBlock
from easyai.model.base_block.cls.senet_block import BottleneckResidualSEBlock
from easyai.model.backbone.utility.registry import REGISTERED_CLS_BACKBONE

__all__ = ['SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'SEResNet152']


class SEResNet(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=BasicResidualSEBlock, num_blocks=(2, 2, 2, 2),
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.SEResNet18)
        self.block = block
        self.num_blocks = num_blocks
        self.out_channels = (64, 128, 256, 516)
        self.strides = (1, 2, 2, 2)
        self.activation_name = activationName
        self.bn_name = bnName
        self.first_output = 64
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        block1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(block1.get_name(), block1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_layer(self.block, num_block, self.out_channels[index],
                            self.strides[index])
            self.in_channel = self.out_channels[index] * self.block.expansion

    def make_layer(self, block, num_block, out_channel, stride):
        down_block = block(self.in_channel, out_channel, stride,
                           bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(down_block.get_name(), down_block,
                            out_channel * block.expansion)
        temp_output_channel = out_channel * block.expansion

        for _ in range(num_block - 1):
            temp_block = block(temp_output_channel, out_channel, 1,
                               bn_name=self.bn_name, activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, temp_output_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SEResNet18)
class SEResNet18(SEResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         block=BasicResidualSEBlock,
                         num_blocks=(2, 2, 2, 2))
        self.set_name(BackboneName.SEResNet18)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SEResNet34)
class SEResNet34(SEResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         block=BasicResidualSEBlock,
                         num_blocks=(3, 4, 6, 3))
        self.set_name(BackboneName.SEResNet34)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SEResNet50)
class SEResNet50(SEResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         block=BottleneckResidualSEBlock,
                         num_blocks=(3, 4, 6, 3))
        self.set_name(BackboneName.SEResNet50)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SEResNet101)
class SEResNet101(SEResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         block=BottleneckResidualSEBlock,
                         num_blocks=(3, 4, 23, 3))
        self.set_name(BackboneName.SEResNet101)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.SEResNet152)
class SEResNet152(SEResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         block=BottleneckResidualSEBlock,
                         num_blocks=(3, 8, 36, 3))
        self.set_name(BackboneName.SEResNet152)

