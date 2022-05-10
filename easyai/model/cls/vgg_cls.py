#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.utility_layer import FcLayer, ActivationLayer
from easyai.model.utility.base_classify_model import BaseClassifyModel
from easyai.model.utility.model_registry import REGISTERED_CLS_MODEL
from torch import nn


@REGISTERED_CLS_MODEL.register_module(ModelName.VggNetCls)
class VggNetCls(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=100):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.VggNetCls)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.Vgg19

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        # avgpool = nn.AdaptiveAvgPool2d((7, 7))

        layer1 = FcLayer(base_out_channels[-1], 4096)
        self.add_block_list(layer1.get_name(), layer1, 4096)

        layer2 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer2.get_name(), layer2, 4096)

        layer3 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, layer3, 4096)

        layer4 = nn.Linear(4096, 4096)
        self.add_block_list(LayerType.FcLinear, layer4, 4096)

        layer5 = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(layer5.get_name(), layer5, 4096)

        layer6 = nn.Dropout()
        self.add_block_list(LayerType.Dropout, layer6, 4096)

        layer7 = nn.Linear(4096, self.class_number)
        self.add_block_list(LayerType.FcLinear, layer7, self.class_number)

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
