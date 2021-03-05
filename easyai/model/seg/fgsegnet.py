#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.utility_layer import RouteLayer, AddLayer, MultiplyLayer
from easyai.model.model_block.base_block.utility.pooling_layer import GlobalAvgPool2d
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock, ConvActivationBlock
from easyai.model.model_block.base_block.utility.upsample_layer import Upsample
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.FgSegNet)
class FgSegNet(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.FgSegNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = BackboneName.MobileNetV2Down4

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = RouteLayer('1')
        output_channel = layer1.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        layer2 = GlobalAvgPool2d()
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        layer3 = RouteLayer('6')
        output_channel = layer3.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer3.get_name(), layer3, output_channel)

        conv1 = ConvBNActivationBlock(self.block_out_channels[-1], 64,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 64)

        conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.add_block_list(LayerType.Convolutional, conv2, 64)

        layer4 = RouteLayer('17')
        output_channel = layer4.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer4.get_name(), layer4, output_channel)

        conv3 = ConvBNActivationBlock(self.block_out_channels[-1], 64,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 64)

        layer5 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(layer5.get_name(), layer5, self.block_out_channels[-1])

        layer6 = AddLayer('-1,-4')
        output_channel = layer6.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer6.get_name(), layer6, output_channel)

        conv4 = ConvBNActivationBlock(output_channel, 16,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv4.get_name(), conv4, 16)

        layer7 = MultiplyLayer('-1,-9')
        output_channel = layer7.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer7.get_name(), layer7, output_channel)

        layer8 = AddLayer('-1,-2')
        output_channel = layer8.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(layer8.get_name(), layer8, output_channel)

        layer9 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(layer9.get_name(), layer9, self.block_out_channels[-1])

        conv5 = ConvBNActivationBlock(output_channel, 64,
                                      kernel_size=3, stride=1, padding=1,
                                      bnName=self.bn_name, activationName=self.activation_name)
        self.add_block_list(conv5.get_name(), conv5, 64)

        if self.class_number == 1:
            conv6 = ConvActivationBlock(self.block_out_channels[-1], self.class_number,
                                        kernel_size=1, stride=1, padding=0,
                                        activationName=ActivationType.Sigmoid)
        else:
            conv6 = ConvActivationBlock(self.block_out_channels[-1], self.class_number,
                                        kernel_size=1, stride=1, padding=0,
                                        activationName=ActivationType.Linear)
        self.add_block_list(conv6.get_name(), conv6, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        if self.class_number == 1:
            loss_config = {'type': LossName.BinaryCrossEntropy2dLoss,
                           'weight_type': 1,
                           'weight': '1, 5',
                           'reduction': 'mean',
                           'ignore_index': 250}
            loss = self.loss_factory.get_loss(loss_config)
        else:
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
            elif LayerType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortRouteLayer in key:
                x = block(layer_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
