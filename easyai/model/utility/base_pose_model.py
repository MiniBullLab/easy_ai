#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model.utility.base_model import *
from easyai.model.model_block.backbone.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BasePoseModel(BaseModel):

    def __init__(self, data_channel, keypoints_number):
        super().__init__(data_channel)
        self.keypoints_number = keypoints_number
        self.backbone_factory = BackboneFactory()
        self.loss_factory = LossFactory()