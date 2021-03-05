#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType
from easyai.model.model_block.base_block.utility.fpn_block import FPNBlock
from easyai.model.utility.base_det_model import *


class KeyPointRCNN(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.KeyPointRCNN)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.ResNet50

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        down_layers = [4, 8, 14, 17]
        down_layer_outputs = [self.block_out_channels[i] if i < 0 else base_out_channels[i]
                              for i in down_layers]
        temp_str = ",".join('%s' % index for index in down_layers)
        fpn_layer = FPNBlock(temp_str, down_layer_outputs, 256)
        self.add_block_list(fpn_layer.get_name(), fpn_layer, 256)

