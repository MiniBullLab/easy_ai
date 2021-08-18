#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.model.cls.patch_core_net import PatchCoreNet
from easyai.model.utility.model_registry import REGISTERED_CLS_MODEL

__all__ = ['OneClassNet']


@REGISTERED_CLS_MODEL.register_module(ModelName.OneClassNet)
class OneClassNet(PatchCoreNet):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.OneClassNet)
