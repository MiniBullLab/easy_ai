#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.model_block.utility.base_block import *
from easyai.model_block.base_block.common.utility_layer import FcLayer, ActivationLayer, NormalizeLayer
from easyai.model_block.utility.block_registry import REGISTERED_MODEL_HEAD


@REGISTERED_MODEL_HEAD.register_module(HeadType.ClassifyHead)
class ClassifyHead(BaseBlock):

    def __init__(self, in_channels, out_channels, class_number,
                 bn_name=NormalizationType.EmptyNormalization,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.ClassifyHead)
        self.fc = FcLayer(in_channels, out_channels)
        self.bn = NormalizeLayer(bn_name, out_channels)
        self.act = ActivationLayer(activation_name)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(out_channels, class_number)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear(x)
        return x


@REGISTERED_MODEL_HEAD.register_module(HeadType.CenterClassifyHead)
class CenterClassifyHead(BaseBlock):

    def __init__(self, in_channels, out_channels, class_number):
        super().__init__(HeadType.CenterClassifyHead)
        self.fc = FcLayer(in_channels, out_channels)
        self.act = ActivationLayer(ActivationType.ReLU)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(out_channels, class_number)

    def forward(self, x):
        output = []
        x = self.fc(x)
        output.append(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear(x)
        output.append(x)
        return output
