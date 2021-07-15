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
        # x = x.view(B, C, H * W)
        # x = x.transpose(1, 2)
        x = x.reshape(B, C, H * W)
        x = x.permute((0, 2, 1))
        return x


class EncoderRNNBlock(BaseBlock):
    def __init__(self, in_channels, hidden_size):
        super().__init__(RNNType.EncoderRNNBlock)
        self.out_channels = hidden_size * 2
        self.lstm = nn.LSTM(in_channels, hidden_size,
                            bidirectional=True, num_layers=2)

    def forward(self, x):
        data_tensor = x.transpose(0, 1)
        x, _ = self.lstm(data_tensor)
        x = x.transpose(0, 1)
        return x


class BidirectionalLSTM(BaseBlock):
    # Inputs hidden units Out
    def __init__(self, in_channels, hidden_size, out_channels):
        super().__init__(RNNType.BidirectionalLSTM)
        self.rnn = nn.LSTM(in_channels, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
