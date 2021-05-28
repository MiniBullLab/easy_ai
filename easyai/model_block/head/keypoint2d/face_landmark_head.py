#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.config.name_manager import HeadType
from easyai.config.name_manager import ActivationType
from easyai.model_block.base_block.utility.utility_block import FcActivationBlock


class FaceLandmarkHead(BaseBlock):

    def __init__(self, input_channle, class_number=3,
                 points_count=68, left_count=39):
        super().__init__(HeadType.FaceLandmarkHead)
        self.fc_ldmk = nn.Linear(input_channle, points_count*2)
        self.fc_ldmk_left = nn.Linear(input_channle, left_count * 2)
        self.fc_ldmk_right = nn.Linear(input_channle, left_count * 2)
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
        left_ldmk = self.fc_ldmk_left(x)
        result.append(left_ldmk)
        right_ldmk = self.fc_ldmk_right(x)
        result.append(right_ldmk)
        conf = self.fc_conf(x)
        result.append(conf)
        gauss = self.fc_gauss(x)
        result.append(gauss)
        left_right_cls = self.fc_cls(x)
        result.append(left_right_cls)
        box = self.fc_box(x)
        result.append(box)
        return result
