#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.torch_utility.torch_freeze_layer import TorchFreezeLayer
from easyai.torch_utility.torch_freeze_bn import TorchFreezeNormalization


class FreezePorcess():

    def __init__(self):
        self.freeze_layer = TorchFreezeLayer()
        self.freeze_normalization = TorchFreezeNormalization()

    def freeze_block(self, model, layer_name, flag=0):
        self.freeze_layer.freeze(model, layer_name, flag)
        self.freeze_layer.print_freeze_layer(model)

    def freeze_bn(self, model, layer_name, flag=0):
        self.freeze_normalization.freeze_normalization_layer(model, layer_name, flag)
