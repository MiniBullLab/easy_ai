#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.model_block.utility.base_block import *
from easyai.model_block.utility.block_registry import REGISTERED_MODEL_HEAD


@REGISTERED_MODEL_HEAD.register_module(HeadType.MultiOutputHead)
class MultiOutputHead(BaseBlock):

    def __init__(self, layers):
        super().__init__(HeadType.MultiOutputHead)
        self.layers = [int(x) for x in layers.split(',') if x.strip()]

    def forward(self, layer_outputs, base_outputs):
        # print(self.layers)
        result = [layer_outputs[i] if i < 0 else base_outputs[i] for i in self.layers]
        # x = torch.cat(temp_layer_outputs, 1)
        return result
