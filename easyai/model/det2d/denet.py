#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.model.det2d.yolov3_det2d import YoloV3Det2d
from easyai.model.utility.model_registry import REGISTERED_DET2D_MODEL


@REGISTERED_DET2D_MODEL.register_module(ModelName.DetNet)
class DetNet(YoloV3Det2d):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.DetNet)
