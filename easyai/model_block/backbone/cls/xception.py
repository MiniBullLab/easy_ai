#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""xception in pytorch
[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.cls.xception_block import EntryFlow, ExitFLow
from easyai.model_block.base_block.cls.xception_block import MiddleFLowBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE

__all__ = ['Xception']


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Xception)
class Xception(BaseBackbone):

    def __init__(self, data_channel=3,
                 block=MiddleFLowBlock,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.Xception)
        self.block = block
        self.block_number = 8
        self.activation_name = activationName
        self.bn_name = bnName

        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        entry_flow = EntryFlow(data_channel=self.data_channel,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(entry_flow.get_name(), entry_flow, 728)

        for _ in range(self.block_number):
            temp_block = self.block()
            self.add_block_list(temp_block.get_name(), temp_block, 728)

        exit_flow = ExitFLow(bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(exit_flow.get_name(), exit_flow, 2048)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list



