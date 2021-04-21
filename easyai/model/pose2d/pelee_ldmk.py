#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType, LayerType, HeadType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.utility_layer import RouteLayer
from easyai.model.model_block.base_block.utility.pooling_layer import MyAvgPool2d
from easyai.model.model_block.head.face_landmark_head import FaceLandmarkHead
from easyai.model.utility.base_pose_model import *
from easyai.model.utility.registry import REGISTERED_POSE2D_MODEL


@REGISTERED_POSE2D_MODEL.register_module(ModelName.PeleeLandmark)
class PeleeLandmark(BasePoseModel):

    def __init__(self, data_channel=3, points_count=68):
        super().__init__(data_channel, points_count)
        self.set_name(ModelName.PeleeLandmark)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.PeleeNetTransition24

        self.loss_config = {"type": LossName.FaceLandmarkLoss,
                            "input_size": "128,128",
                            "points_count": points_count,
                            "wing_w": 15,
                            "wing_e": 3,
                            "gaussian_scale": 4,
                            "ignore_value": -1}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        route1 = RouteLayer("11")
        output_channel1 = route1.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route1.get_name(), route1, output_channel1)

        avg_pool1 = MyAvgPool2d(kernel_size=8)
        self.add_block_list(avg_pool1.get_name(), avg_pool1, output_channel1)

        route2 = RouteLayer("19")
        output_channel2 = route2.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route2.get_name(), route2, output_channel2)

        avg_pool2 = MyAvgPool2d(kernel_size=4)
        self.add_block_list(avg_pool2.get_name(), avg_pool2, output_channel2)

        route3 = RouteLayer("23")
        output_channel3 = route3.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route3.get_name(), route3, output_channel3)

        avg_pool3 = MyAvgPool2d(kernel_size=4)
        self.add_block_list(avg_pool3.get_name(), avg_pool3, output_channel3)

        route4 = RouteLayer("-1,-3,-5")
        output_channel4 = route4.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route4.get_name(), route4, output_channel4)

        landmark = FaceLandmarkHead(output_channel4, 3, self.points_count)
        self.add_block_list(landmark.get_name(), landmark, self.points_count*2)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = list()
        layer_outputs = list()
        multi_output = list()
        output = list()
        output.append(x.clone())
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif HeadType.FaceLandmarkHead in key:
                x = block(x)
                multi_output.extend(x)
            elif self.loss_factory.has_loss(key):
                output.extend(multi_output)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
