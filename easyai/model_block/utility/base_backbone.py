#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.model_block.utility.abstract_model import *


class BaseBackbone(AbstractModel):

    def __init__(self, data_channel):
        super().__init__()
        self.data_channel = data_channel

    def get_outchannel_list(self):
        return self.block_out_channels

    def get_data_channel(self):
        return self.data_channel
