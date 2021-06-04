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
        assert H == 1
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x


class EncoderRNNBlock(BaseBlock):
    def __init__(self, in_channels, hidden_size):
        super().__init__(RNNType.EncoderRNNBlock)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size,
                            bidirectional=True, num_layers=2,
                            batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
