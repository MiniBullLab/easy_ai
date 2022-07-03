#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.base_model import *

from easy_pc.model_block.utility.pc_backbone_factory import PCBackboneFactory
from easy_pc.loss.utility.pc_loss_factory import PCLossFactory


class BasePCClassifyModel(BaseModel):

    def __init__(self, data_channel, class_number):
        super().__init__(data_channel)
        assert class_number > 0
        self.class_number = class_number
        self.backbone_factory = PCBackboneFactory()
        self.loss_factory = PCLossFactory()
