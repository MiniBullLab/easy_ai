#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import HeadType
from easyai.model.model_block.base_block.utility.base_block import *


class SSDBoxHead(BaseBlock):

    def __init__(self, input_channle, anchor_number, class_number,
                 kernel_size=3, is_gaussian=False):
        super().__init__(HeadType.SSDBoxHead)
        self.anchor_number = anchor_number
        self.class_number = class_number
        self.loc_output = 4
        # Just for exp convenience
        if is_gaussian:
            self.loc_output += 4
        pad = ((kernel_size - 1) // 2)
        self.loc_layer = nn.Conv2d(input_channle, anchor_number * self.loc_output,
                                   kernel_size=kernel_size, padding=pad, stride=1)
        self.cls_layer = nn.Conv2d(input_channle, anchor_number * class_number,
                                   kernel_size=kernel_size, padding=pad, stride=1)

    def forward(self, x):
        loc_cls_result = []
        y_loc = self.loc_layer(x)
        y_cls = self.cls_layer(x)
        loc_cls_result.append(y_loc)
        loc_cls_result.append(y_cls)
        return loc_cls_result


class MultiSSDBoxHead(BaseBlock):

    def __init__(self, input_channels, anchor_numbers, class_number,
                 kernel_size=3, is_gaussian=False):
        super().__init__(HeadType.MultiSSDBoxHead)
        self.anchor_number_list = anchor_numbers
        self.class_number = class_number
        self.loc_output = 4
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        # Just for exp convenience
        if is_gaussian:
            self.loc_output += 4
        pad = ((kernel_size - 1) // 2)
        for i in range(len(input_channels)):
            self.loc_layers += [
                nn.Conv2d(
                    input_channels[i],
                    self.anchor_number_list[i] * self.loc_output,
                    kernel_size=kernel_size,
                    padding=pad)
            ]
            self.conf_layers += [
                nn.Conv2d(
                    input_channels[i],
                    self.anchor_number_list[i] * self.class_number,
                    kernel_size=kernel_size,
                    padding=pad)
            ]

    def forward(self, x):
        loc_cls_result = []
        for index, (x, loc, conf) in enumerate(zip(x, self.loc_layers, self.conf_layers)):
            y_loc = self.loc(x)
            y_cls = self.conf(x)
            loc_cls_result.append(y_loc)
            loc_cls_result.append(y_cls)
        return loc_cls_result
