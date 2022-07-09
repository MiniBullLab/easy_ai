#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.base_model import *
from easyai.model_block.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BaseReIDModel(BaseModel):

    def __init__(self, data_channel, class_number, reid, max_id=-1):
        super().__init__(data_channel)
        self.class_number = class_number
        self.reid = reid
        self.max_id = max_id
        self.backbone_factory = BackboneFactory()
        self.loss_factory = LossFactory()