#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType
from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.common.utility_layer import FcLayer, ActivationLayer
from easyai.model_block.base_block.common.utility_block import FcActivationBlock
from easyai.model_block.base_block.common.pooling_layer import MultiROIPooling
from easyai.model_block.utility.base_block import *


class MultiROIBoxHead(BaseBlock):

    def __init__(self, in_channels, out_channels, class_number,
                 pool_resolution, pool_scales, pool_sampling_ratio,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.MultiROIBoxHead)
        self.pooling = MultiROIPooling(out_channels=in_channels,
                                       output_size=(pool_resolution, pool_resolution),
                                       scales=pool_scales,
                                       sampling_ratio=pool_sampling_ratio)
        input_size = in_channels * pool_resolution ** 2
        self.fc1 = FcLayer(input_size, out_channels)
        self.activate = ActivationLayer(activation_name=activation_name)
        self.fc2 = FcActivationBlock(out_channels, out_channels,
                                     activationName=activation_name)

        self.cls_score = nn.Linear(out_channels, class_number)
        self.bbox_pred = nn.Linear(out_channels, class_number * 4)

    def forward(self, x, proposals):
        output = []
        x = self.pooler(x, proposals)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        class_logits = self.cls_score(x)
        output.append(class_logits)
        box_regression = self.bbox_pred(x)
        output.append(box_regression)
        return tuple(output)
