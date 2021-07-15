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
        data_tensor = x.transpose(0, 1)
        recurrent, _ = self.rnn(data_tensor)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        output = output.transpose(0, 1)
        return output


class RNNAttention(BaseBlock):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__(RNNType.RNNAttention)
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
