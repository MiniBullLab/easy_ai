#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NeckType
from easyai.model_block.base_block.rnn.rnn_block import Im2SeqBlock
from easyai.model_block.base_block.rnn.rnn_block import EncoderRNNBlock
from easyai.model_block.utility.base_block import *


class SequenceEncoder(BaseBlock):
    def __init__(self, in_channels, hidden_size=48, rnn_reshape=False):
        super().__init__(NeckType.SequenceEncoder)
        self.encoder_reshape = Im2SeqBlock(in_channels)
        self.encoder = EncoderRNNBlock(in_channels, hidden_size,
                                       use_reshape=rnn_reshape)
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        x = self.encoder_reshape(x)
        x = self.encoder(x)
        return x
