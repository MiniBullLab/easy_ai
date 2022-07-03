#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock

from easy_pc.name_manager.pc_backbone_name import PointCloudBackboneName
from easy_pc.model_block.utility.pc_block_registry import REGISTERED_PC_DET3D_BACKBONE

__all__ = ['SecondNet']


@REGISTERED_PC_DET3D_BACKBONE.register_module(PointCloudBackboneName.SecondNet)
class SecondNet(BaseBackbone):

    def __init__(self, data_channel=64, num_blocks=(3, 5, 5),
                 out_channels=(64, 128, 256), strides=(2, 2, 2),
                 bn_name=NormalizationType.BatchNormalize2d,
                 act_name=ActivationType.ReLU):
        super().__init__(data_channel)
        assert len(num_blocks) == len(out_channels)
        assert len(out_channels) == len(strides)
        self.set_name(PointCloudBackboneName.SecondNet)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.bn_name = bn_name
        self.act_name = act_name

        self.in_channels = data_channel

        self.create_block_list()

    def create_block_list(self):
        for index, num_block in enumerate(self.num_blocks):
            self.make_blocks(self.out_channels[index], self.num_blocks[index],
                             self.strides[index], self.bn_name, self.act_name)
            self.in_channels = self.block_out_channels[-1]

    def make_blocks(self, out_channels, num_blocks, stride,
                    bn_name, act_name):
        down_layers = ConvBNActivationBlock(in_channels=self.in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            padding=1,
                                            stride=stride,
                                            bnName=bn_name,
                                            activationName=act_name)
        name = "down_%s" % down_layers.get_name()
        self.add_block_list(name, down_layers, out_channels)
        for j in range(num_blocks):
            temp_layers = ConvBNActivationBlock(in_channels=out_channels,
                                                out_channels=out_channels,
                                                kernel_size=3,
                                                padding=1,
                                                stride=1,
                                                bnName=bn_name,
                                                activationName=act_name)
            self.add_block_list(temp_layers.get_name(), temp_layers, out_channels)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list
