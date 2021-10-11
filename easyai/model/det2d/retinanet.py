#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
Focal Loss for Dense Object Detection
"""

from easyai.name_manager import ModelName
from easyai.name_manager import NormalizationType, ActivationType
from easyai.model_block.utility.base_model import *
from easyai.model_block.backbone.utility import BackboneFactory


class RetinaNet(BaseModel):

    def __init__(self, data_channel=3, class_num=80):
        super().__init__()
        self.set_name(ModelName.FastSCNN)
        self.data_channel = data_channel
        self.class_number = class_num
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.factory = BackboneFactory()
        self.create_block_list()