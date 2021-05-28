#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.config.name_manager import ModelName
from easyai.config.name_manager import BackboneName
from easyai.config.name_manager import NormalizationType, ActivationType
from easyai.config.name_manager import LayerType, BlockType
from easyai.config.name_manager import LossName
from easyai.model_block.base_block.utility.utility_layer import FcLayer
from easyai.model_block.base_block.utility.utility_layer import NormalizeLayer, ActivationLayer
from easyai.model.utility.base_classify_model import BaseClassifyModel
from easyai.model.utility.registry import REGISTERED_CLS_MODEL
from torch import nn


@REGISTERED_CLS_MODEL.register_module(ModelName.GhostNetCls)
class GhostNetCls(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=100):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.GhostNetCls)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.GhostNet

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.add_block_list(LayerType.GlobalAvgPool, avgpool, base_out_channels[-1])

        output_channel = 1280
        layer1 = FcLayer(base_out_channels[-1], output_channel)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        layer2 = NormalizeLayer(bn_name=NormalizationType.BatchNormalize1d,
                                out_channel=output_channel)
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        layer3 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        layer4 = nn.Dropout(0.2)
        self.add_block_list(LayerType.Dropout, layer4, output_channel)

        layer5 = nn.Linear(output_channel, self.class_number)
        self.add_block_list(LayerType.FcLinear, layer5, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.CrossEntropy2dLoss,
                       'weight_type': 0,
                       'reduction': 'mean',
                       'ignore_index': 250}
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
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
