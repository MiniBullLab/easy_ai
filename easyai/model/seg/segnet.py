#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.base_name.model_name import ModelName
from easyai.model.seg.fgsegnet import FgSegNet
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.SegNet)
class SegNet(FgSegNet):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, 1)
        self.set_name(ModelName.SegNet)
