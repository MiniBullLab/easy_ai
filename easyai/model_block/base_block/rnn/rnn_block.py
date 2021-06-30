#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import RNNType
from easyai.model_block.utility.base_block import *


class Im2SeqBlock(BaseBlock):
    def __init__(self, in_channels):
        super().__init__(RNNType.Im2SeqBlock)
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == 1
        x = x.view(B, C, H * W)
        x = x.transpose(1, 2)
        return x


class EncoderRNNBlock(BaseBlock):
    def __init__(self, in_channels, hidden_size, use_reshape=False):
        super().__init__(RNNType.EncoderRNNBlock)
        self.use_reshape = use_reshape
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size,
                            bidirectional=True, num_layers=2)

    def forward(self, x):
        if self.use_reshape:
            data_tensor = x.transpose(0, 1)
            x, _ = self.lstm(data_tensor)
            x = x.transpose(0, 1)
        else:
            x, _ = self.lstm(x)
        return x
