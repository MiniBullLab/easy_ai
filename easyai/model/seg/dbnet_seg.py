#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""DBNet
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  title={Real-time Scene Text Detection with Differentiable Binarization},
  booktitle={Proc. AAAI},
  year={2020}
"""

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import BlockType, NeckType, HeadType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.neck.db_fpn_neck import DBFPNNeck
from easyai.model_block.head.seg.db_head import DBHead
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_SEG_MODEL


class DBNet(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.DBNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = BackboneName.ResNet18
        self.feature_out_channels = 256
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        down_layers = [3, 5, 7, 9]
        down_layer_outputs = [self.block_out_channels[i] if i < 0 else base_out_channels[i]
                              for i in down_layers]
        temp_str = ",".join('%s' % index for index in down_layers)
        fpn_layer = DBFPNNeck(temp_str, down_layer_outputs, self.feature_out_channels)
        self.add_block_list(fpn_layer.get_name(), fpn_layer, 256 // 4 * 4)

        head_layer = DBHead(in_channels=self.block_out_channels[-1],
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(head_layer.get_name(), head_layer, 1)

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
            elif NeckType.DBFPNNeck in key:
                x = block(layer_outputs, base_outputs)
            elif HeadType.DBHead in key:
                x = block(x)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
