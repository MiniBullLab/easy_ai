#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.base_model import *
from easyai.model_block.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BasePoseModel(BaseModel):

    def __init__(self, data_channel, points_count):
        super().__init__(data_channel)
        self.points_count = points_count
        self.backbone_factory = BackboneFactory()
        self.loss_factory = LossFactory()