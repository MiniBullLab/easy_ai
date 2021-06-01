#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.upsample_layer import Upsample
from easyai.model_block.base_block.common.utility_layer import RouteLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.cls.deeplab_block import ASPPBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.DeepLabV3)
class DeepLabV3(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=19):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.DeepLabV3)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = BackboneName.ResNet50
        self.create_block_list()
        self.loss_flag = -1

    def create_block_list(self):
        self.clear_list()
        self.lossList = []

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        head = ASPPBlock(2048, bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(head.get_name(), head, 512)

        out_conv = nn.Conv2d(512, self.class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.add_block_list(LayerType.Convolutional, out_conv, self.class_number)

        up = Upsample(scale_factor=32, mode='bilinear')
        self.add_block_list(up.get_name(), up, self.block_out_channels[-1])

        self.loss_flag = 0
        self.clear_list()

        self.dsn_head(base_out_channels)

        self.loss_flag = 1
        self.clear_list()

        self.create_loss_list()

    def dsn_head(self, base_out_channels):
        route = RouteLayer('14')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in route.layers])
        self.add_block_list(route.get_name(), route, output_channel)

        conv1 = ConvBNActivationBlock(in_channels=output_channel,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=True,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        dropout = nn.Dropout2d(0.1)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        out_conv = nn.Conv2d(512, self.class_number, kernel_size=1, stride=1, padding=0, bias=True)
        self.add_block_list(LayerType.Convolutional, out_conv, self.class_number)

        up1 = Upsample(scale_factor=16, mode='bilinear')
        self.add_block_list(up1.get_name(), up1, self.block_out_channels[-1])

    def create_loss_list(self, input_dict=None):
        if self.loss_flag == 0:
            loss_config = {'type': LossName.CrossEntropy2dLoss,
                           'weight_type': 0,
                           'reduction': 'mean',
                           'ignore_index': 250}
            loss = self.loss_factory.get_loss(loss_config)
            self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
            self.lossList.append(loss)
        elif self.loss_flag == 1:
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
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output
