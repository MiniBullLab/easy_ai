#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.base_model import *
from easyai.model_block.utility.backbone_factory import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory

from easy_pc.model_block.utility.pc_backbone_factory import PCBackboneFactory
from easy_pc.loss.utility.pc_loss_factory import PCLossFactory


class BasePCDet3dModel(BaseModel):

    def __init__(self, data_channel, class_number):
        super().__init__(data_channel)
        assert class_number > 0
        self.class_number = class_number

        self.pc_backbone_factory = PCBackboneFactory()
        self.pc_loss_factory = PCLossFactory()

        self.image_backbone_factory = BackboneFactory()
        self.image_loss_factory = LossFactory()
