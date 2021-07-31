#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.model.rnn.crnn import CRNN
from easyai.model.rnn.cnn_ctc import CNNCTC
from easyai.model.utility.model_registry import REGISTERED_RNN_MODEL

__all__ = ['TextNet']


@REGISTERED_RNN_MODEL.register_module(ModelName.TextNet)
class TextNet(CNNCTC):

    def __init__(self, data_channel=3, class_number=100):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.TextNet)
