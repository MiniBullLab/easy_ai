#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.model_block.utility.base_block import *


class YoloBoxHead(BaseBlock):

    def __init__(self, layers, input_channels,
                 class_number, anchor_count, kernel_size=1):
        super().__init__(HeadType.YoloBoxHead)
        self.layers = [int(x) for x in layers.split(',') if x.strip()]
        assert len(self.layers) == len(input_channels)
        self.layers_count = len(self.layers)
        self.anchor_count = anchor_count
        self.class_number = class_number
        self.no = class_number + 5  # number of outputs per anchor
        self.block = nn.ModuleList(nn.Conv2d(x, self.no * self.anchor_count, kernel_size) for x in input_channels)

    def forward(self, layer_outputs, base_outputs):
        result = []
        x_list = [layer_outputs[i] if i < 0 else base_outputs[i] for i in self.layers]
        for i in range(self.layers_count):
            result.append(self.block[i](x_list[i]))
        return result
