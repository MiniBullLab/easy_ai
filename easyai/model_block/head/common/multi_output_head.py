#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager import HeadType


class MultiOutputHead(BaseBlock):

    def __init__(self, layers):
        super().__init__(HeadType.MultiOutputHead)
        self.layers = [int(x) for x in layers.split(',') if x.strip()]

    def forward(self, layer_outputs, base_outputs):
        # print(self.layers)
        result = [layer_outputs[i] if i < 0 else base_outputs[i] for i in self.layers]
        # x = torch.cat(temp_layer_outputs, 1)
        return result
