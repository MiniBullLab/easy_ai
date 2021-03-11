#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model.utility.base_model import *
from easyai.model.model_block.backbone.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BaseDetectionModel(BaseModel):

    def __init__(self, data_channel, class_number):
        super().__init__(data_channel)
        self.class_number = class_number
        self.backbone_factory = BackboneFactory()
        self.loss_factory = LossFactory()
