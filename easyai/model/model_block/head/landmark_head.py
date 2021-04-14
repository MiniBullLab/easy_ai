#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.base_name.block_name import HeadType
from easyai.base_name.block_name import ActivationType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.model.model_block.base_block.utility.utility_block import FcActivationBlock


class LandmarkHead(BaseBlock):

    def __init__(self, input_channle, class_number=3, points_count=68):
        super().__init__(HeadType.LandmarkHead)
        self.fc_ldmk = nn.Linear(input_channle, points_count*2)
        self.fc_cls = nn.Linear(input_channle, class_number)
        self.fc_box = nn.Linear(input_channle, 4)
        self.fc_conf = FcActivationBlock(input_channle, points_count,
                                         activationName=ActivationType.Sigmoid)
        self.fc_gauss = FcActivationBlock(input_channle, points_count*2,
                                          activationName=ActivationType.Sigmoid)

    def forward(self, x):
        result = []
        ldmk = self.fc_ldmk(x)
        result.append(ldmk)
        conf = self.fc_conf(x)
        result.append(conf)
        gauss = self.fc_gauss(x)
        result.append(gauss)
        left_right_cls = self.fc_cls(x)
        result.append(left_right_cls)
        box = self.fc_box(x)
        result.append(box)
        return result
