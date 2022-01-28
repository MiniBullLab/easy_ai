#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType, HeadType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.pooling_layer import MyAvgPool2d
from easyai.model_block.head.cls.classify_head import ClassifyHead
from easyai.model.utility.base_reid_model import *
from easyai.model.utility.model_registry import REGISTERED_REID_MODEL


@REGISTERED_REID_MODEL.register_module(ModelName.DeepSortNet)
class DeepSortNet(BaseReIDModel):

    def __init__(self, data_channel=3, class_number=751, reid=0):
        super().__init__(data_channel, class_number, reid)
        self.set_name(ModelName.DeepSortNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.DeepSortBackbone

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        avgpool = MyAvgPool2d((8, 4), 1)
        self.add_block_list(avgpool.get_name(), avgpool, self.block_out_channels[-1])

        head = ClassifyHead(512, 256, self.class_number,
                            bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(head.get_name(), head, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        if self.class_number > 1:
            loss_config = {'type': LossName.CrossEntropy2dLoss,
                           'weight_type': 0,
                           'reduction': 'mean',
                           'ignore_index': 250}
            loss = self.loss_factory.get_loss(loss_config)
            self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
            self.lossList.append(loss)
        else:
            self.lossList = []
            loss_config = {'type': LossName.BinaryCrossEntropy2dLoss,
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
            elif HeadType.ClassifyHead in key:
                if self.reid > 0:
                    x = x.div(x.norm(p=2, dim=1, keepdim=True))
                    # print("reid:", x.shape)
                else:
                    x = block(x)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
