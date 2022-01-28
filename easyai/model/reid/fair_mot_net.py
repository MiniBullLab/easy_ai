#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType, HeadType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.head.reid.fair_mot_head import FairMOTHead
from easyai.model.utility.base_reid_model import BaseReIDModel
from easyai.model.utility.model_registry import REGISTERED_REID_MODEL


@REGISTERED_REID_MODEL.register_module(ModelName.FairMOTNet)
class FairMOTNet(BaseReIDModel):

    def __init__(self, data_channel=3, class_number=1, reid=128):
        super().__init__(data_channel, class_number, reid)
        self.set_name(ModelName.FairMOTNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.Swish

        self.model_args['type'] = BackboneName.Yolov5s_Backbone

        self.loss_config = {"type": LossName.EmptyLoss}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        head = FairMOTHead(64, (self.class_number, self.reid, 2, 4),
                           self.activation_name)
        self.add_block_list(head.get_name(), head, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
        self.add_block_list(loss.get_name(), loss, -1)
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
            elif HeadType.FairMOTHead in key:
                x = block(layer_outputs, base_outputs)
            elif self.loss_factory.has_loss(key):
                output.extend(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
