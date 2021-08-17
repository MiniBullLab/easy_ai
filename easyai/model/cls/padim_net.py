#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
Original Paper : Thomas Defard, Aleksandr Setkov, Angelique Loesch, Romaric Audigier.
PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization
"""

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType, HeadType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.head.cls.patch_head import PadimHead
from easyai.model.utility.base_classify_model import BaseClassifyModel
from easyai.model.utility.model_registry import REGISTERED_CLS_MODEL

__all__ = ['PadimNet']


@REGISTERED_CLS_MODEL.register_module(ModelName.PadimNet)
class PadimNet(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.PadimNet)
        self.class_number = 1
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.WideResnet50V2

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        head = PadimHead("4,8,14")
        self.add_block_list(head.get_name(), head, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.EmptyLoss}
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
            elif HeadType.PadimHead in key:
                x = block(layer_outputs, base_outputs)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output

