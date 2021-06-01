#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.name_manager import ActivationType
from easyai.name_manager import HeadType
from easyai.model_block.base_block.utility.utility_block import ConvActivationBlock


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
        self.cls_logits = ConvActivationBlock(in_channels=input_channle,
                                              out_channels=anchor_number,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              activationName=ActivationType.Sigmoid)

        self.bbox_pred = nn.Conv2d(input_channle, anchor_number * 4,
                                   kernel_size=1, stride=1)

    def forward(self, x):
        output = []
        for feature in x:
            t = self.conv(feature)
            output.append(self.cls_logits(t))
            output.append(self.bbox_pred(t))
        return tuple(output)

