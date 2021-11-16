#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.model.seg.dbnet_seg import DBNet
from easyai.model.utility.model_registry import REGISTERED_DET2D_MODEL


@REGISTERED_DET2D_MODEL.register_module(ModelName.OCRDetNet)
class OCRDetNet(DBNet):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.OCRDetNet)
