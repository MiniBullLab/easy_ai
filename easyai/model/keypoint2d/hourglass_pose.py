#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager import ModelName
from easyai.name_manager import BackboneName
from easyai.name_manager import NormalizationType, ActivationType
from easyai.name_manager import BlockType, LayerType, HeadType
from easyai.name_manager import LossName
from easyai.model_block.base_block.utility.utility_layer import RouteLayer
from easyai.model_block.base_block.utility.pooling_layer import MyAvgPool2d
from easyai.model_block.head.utility import MultiOutputHead
from easyai.model.utility.base_pose_model import *
from easyai.model.utility.model_registry import REGISTERED_KEYPOINT2D_MODEL


@REGISTERED_KEYPOINT2D_MODEL.register_module(ModelName.HourglassPose)
class HourglassPose(BasePoseModel):

    def __init__(self, data_channel=3, points_count=68):
        super().__init__(data_channel, points_count)
        self.set_name(ModelName.HourglassPose)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.loss_config = {"type": LossName.JointsMSELoss,
                            "input_size": "128,128",
                            "reduction": 4,
                            "points_count": points_count}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.model_args['type'] = BackboneName.HourGlassNet
        self.model_args['final_out_channel'] = self.points_count
        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        head = MultiOutputHead("8,17,26")
        self.add_block_list(head.get_name(), head, self.points_count)

        route = RouteLayer("25")
        output_channel = route.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route.get_name(), route, output_channel)
        conv = nn.Conv2d(output_channel, output_channel,
                         kernel_size=3, stride=2, padding=0)
        self.add_block_list(LayerType.Convolutional, conv, output_channel)

        avg_pool = MyAvgPool2d(15)
        self.add_block_list(avg_pool.get_name(), avg_pool, output_channel)

        fc = nn.Linear(output_channel, self.points_count*2)
        self.add_block_list(LayerType.FcLinear, fc, self.points_count*2)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        multi_output = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif HeadType.MultiOutputHead in key:
                temp_outputs = block(layer_outputs, base_outputs)
                multi_output.extend(temp_outputs)
                x = temp_outputs[-1]
            elif self.loss_factory.has_loss(key):
                output.append(x)
                output.extend(multi_output)
            else:
                x = block(x)
            print(key, x.shape)
            layer_outputs.append(x)
        return output
