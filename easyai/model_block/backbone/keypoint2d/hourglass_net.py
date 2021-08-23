#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_layer import AddLayer, RouteLayer
from easyai.model_block.base_block.common.pooling_layer import MyAvgPool2d
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.residual_block import ResidualV2Block
from easyai.model_block.base_block.keypoint2d.hourglass_block import HourglassBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.HourGlassNet)
class HourGlassNet(BaseBackbone):

    def __init__(self, data_channel=3, stacks_count=3, depth=4,
                 feature_channel=96, final_out_channel=68,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.HourGlassNet)
        self.stacks_count = stacks_count
        self.depth = depth
        self.feature_channel = feature_channel
        self.final_out_channel = final_out_channel
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.first_output = 32

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=self.first_output,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       dilation=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        layer2 = MyAvgPool2d(kernel_size=2, stride=2,
                             padding=0, ceil_mode=False,
                             use_reshape=False)
        self.add_block_list(layer2.get_name(), layer2, self.first_output)

        layer3 = ResidualV2Block(1, self.first_output, 24,
                                 stride=1, expansion=2,
                                 bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, 24*2)

        layer4 = ResidualV2Block(1, 24*2, 24,
                                 stride=1, expansion=2,
                                 bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(layer4.get_name(), layer4, 24 * 2)

        layer5 = ResidualV2Block(1, 24 * 2, self.feature_channel // 2,
                                 stride=1, expansion=2,
                                 bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(layer5.get_name(), layer5, self.feature_channel)

        for hg_module in range(self.stacks_count):
            hourglass_block = HourglassBlock(self.depth, self.feature_channel,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
            self.add_block_list(hourglass_block.get_name(), hourglass_block, self.feature_channel)
            temp_layer1 = ResidualV2Block(1, self.feature_channel, self.feature_channel // 2,
                                          stride=1, expansion=2,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
            self.add_block_list(temp_layer1.get_name(), temp_layer1, self.feature_channel)
            temp_layer2 = ConvBNActivationBlock(in_channels=self.feature_channel,
                                                out_channels=self.feature_channel,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False,
                                                bnName=self.bn_name,
                                                activationName=self.activation_name)
            self.add_block_list(temp_layer2.get_name(), temp_layer2, self.feature_channel)
            conv1 = nn.Conv2d(self.feature_channel, self.final_out_channel,
                              kernel_size=1, stride=1, padding=0)
            self.add_block_list(LayerType.Convolutional, conv1, self.final_out_channel)

            if hg_module < self.stacks_count - 1:
                route1 = RouteLayer("-2")
                self.add_block_list(route1.get_name(), route1, self.feature_channel)
                conv2 = nn.Conv2d(self.feature_channel, self.feature_channel,
                                  kernel_size=1, stride=1, padding=0)
                self.add_block_list(LayerType.Convolutional, conv2, self.feature_channel)
                route2 = RouteLayer("-3")
                self.add_block_list(route2.get_name(), route2, self.final_out_channel)
                conv3 = nn.Conv2d(self.final_out_channel, self.feature_channel,
                                  kernel_size=1, stride=1, padding=0)
                self.add_block_list(LayerType.Convolutional, conv3, self.feature_channel)
                add_layer = AddLayer("-1,-3,-9")
                self.add_block_list(add_layer.get_name(), add_layer, self.feature_channel)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        for key, block in self._modules.items():
            if LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return layer_outputs
