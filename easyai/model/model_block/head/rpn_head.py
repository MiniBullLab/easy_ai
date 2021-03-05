#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.base_name.block_name import ActivationType
from easyai.base_name.block_name import HeadType
from easyai.model.model_block.base_block.utility.utility_block import ConvActivationBlock
from easyai.model.model_block.base_block.utility.base_block import *


class MultiRPNHead(BaseBlock):

    def __init__(self, input_channle, anchor_number,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.MultiRPNHead)

        self.conv = ConvActivationBlock(in_channels=input_channle,
                                        out_channels=input_channle,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        activationName=activation_name)
        self.cls_logits = nn.Conv2d(input_channle, anchor_number,
                                    kernel_size=1, stride=1)

        self.bbox_pred = nn.Conv2d(input_channle, anchor_number * 4,
                                   kernel_size=1, stride=1)

    def forward(self, x):
        output = []
        for feature in x:
            t = self.conv(feature)
            output.append(self.cls_logits(t))
            output.append(self.bbox_pred(t))
        return tuple(output)

