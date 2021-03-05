#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.base_model import *
from easyai.model.model_block.backbone.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BaseClassifyModel(BaseModel):

    def __init__(self, data_channel, class_number):
        super().__init__(data_channel)
        assert class_number > 0
        self.class_number = class_number
        self.backbone_factory = BackboneFactory()
        self.loss_factory = LossFactory()
