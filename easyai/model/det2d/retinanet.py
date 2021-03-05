#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
Focal Loss for Dense Object Detection
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.model.utility.base_model import *
from easyai.model.model_block.backbone.utility.backbone_factory import BackboneFactory


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