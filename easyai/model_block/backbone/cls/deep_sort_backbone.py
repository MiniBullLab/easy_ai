#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.cls.deep_sort_blcok import BasicBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


__all__ = ['DeepSortBackbone']


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.DeepSortBackbone)
class DeepSortBackbone(BaseBackbone):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.DeepSortBackbone)
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.first_output = 64

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=True,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        # 32 64 32
        self.make_layers(64, 64, 2, False)
        # 32 64 32
        self.make_layers(64, 128, 2, True)
        # 64 32 16
        self.make_layers(128, 256, 2, True)
        # 128 16 8
        self.make_layers(256, 512, 2, True)

    def make_layers(self, c_in, c_out, repeat_times, is_downsample=False):
        for i in range(repeat_times):
            if i == 0:
                temp1 = BasicBlock(c_in, c_out, is_downsample=is_downsample)
                self.add_block_list(temp1.get_name(), temp1, c_out)
            else:
                temp1 = BasicBlock(c_out, c_out)
                self.add_block_list(temp1.get_name(), temp1, c_out)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


