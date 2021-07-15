#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.name_manager.block_name import ActivationType
from easyai.model_block.utility.base_block import *
from easyai.model_block.base_block.common.utility_layer import FcLayer, ActivationLayer


class ClassifyHead(BaseBlock):

    def __init__(self, in_channels, out_channels, class_number):
        super().__init__(HeadType.ClassifyHead)
        self.fc = FcLayer(in_channels, out_channels)
        self.act = ActivationLayer(ActivationType.ReLU)
        self.drop = nn.Dropout(0.5)
        self.linear = nn.Linear(out_channels, class_number)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear(x)
        return x
