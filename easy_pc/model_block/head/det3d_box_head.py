#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.base_block import *

from easy_pc.name_manager.pc_block_name import PCHeadType


class Detection3dBoxHead(BaseBlock):

    def __init__(self, input_channle, anchor_number,
                 class_number, code_size=7,
                 kernel_size=1, use_direction_classifier=True):
        super().__init__(PCHeadType.Detection3dBoxHead)
        self.anchor_number = anchor_number
        self.class_number = class_number
        self.code_size = code_size
        self.use_direction_classifier = use_direction_classifier
        self.conv_cls = nn.Conv2d(input_channle,
                                  self.anchor_number * self.class_number,
                                  kernel_size)
        self.conv_reg = nn.Conv2d(input_channle,
                                  self.num_anchors * self.code_size,
                                  kernel_size)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(input_channle,
                                          self.anchor_number * 2,
                                          kernel_size)

    def forward(self, x):
        result = []
        cls_score = self.conv_cls(x)
        result.append(cls_score)
        bbox_pred = self.conv_reg(x)
        result.append(bbox_pred)
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        else:
            dir_cls_preds = None
        result.append(dir_cls_preds)
        return result
