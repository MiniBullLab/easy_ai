#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
Implementation of 'Scene Text Recognition with Sliding Convolutional Character Models'
([pdf](https://arxiv.org/pdf/1709.01727))
"""

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.head.cls.classify_head import ClassifyHead
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_RNN_MODEL

__all__ = ['CNNCTC']


@REGISTERED_RNN_MODEL.register_module(ModelName.CNNCTC)
class CNNCTC(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=6625):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.CRNN)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args["data_channel"] = data_channel * 3
        self.model_args['type'] = BackboneName.TextVGG

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        classifier = ClassifyHead(4 * base_out_channels[-1], 900, self.class_number)
        self.add_block_list(classifier.get_name(), classifier, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.CTCLoss,
                       'blank_index': 0,
                       'reduction': 'mean'}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def single_forward(self, x):
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
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
        return output

    def forward(self, x):
        result = []
        output = []
        for s in range(x.shape[1]):  # x: batch, window, slice channel, h, w
            result.extend(self.single_forward(x[:, s, :, :, :]))
        out = torch.stack(result, axis=0)
        out = out.permute(1, 0, 2)
        output.append(out)
        return output
