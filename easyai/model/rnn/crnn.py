#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType, RNNType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.pooling_layer import MyMaxPool2d
from easyai.model_block.base_block.rnn.rnn_block import Im2SeqBlock
from easyai.model_block.base_block.rnn.rnn_block import EncoderRNNBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_RNN_MODEL

__all__ = ['CRNN']


@REGISTERED_RNN_MODEL.register_module(ModelName.CRNN)
class CRNN(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=6625):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.CRNN)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.TextResNet34

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        pool = MyMaxPool2d(kernel_size=2, stride=2)
        self.add_block_list(pool.get_name(), pool, base_out_channels[-1])

        seq_layer = Im2SeqBlock(self.block_out_channels[-1])
        self.add_block_list(seq_layer.get_name(), seq_layer, seq_layer.out_channels)

        # neck = EncoderRNNBlock(self.block_out_channels[-1], 256)
        # self.add_block_list(neck.get_name(), neck, neck.out_channels)

        layer = nn.Linear(self.block_out_channels[-1], self.class_number)
        self.add_block_list(LayerType.FcLinear, layer, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.CTCLoss,
                       'blank_index': 0,
                       'reduction': 'none',
                       'use_weight': True,
                       'use_focal': False}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            # elif RNNType.Im2SeqBlock in key:
            #     x = block(layer_outputs)
            #     output.append(x)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
