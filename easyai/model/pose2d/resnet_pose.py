#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType, LayerType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.upsample_layer import DeConvBNActivationBlock
from easyai.model.utility.base_pose_model import *
from easyai.model.utility.registry import REGISTERED_POSE2D_MODEL


@REGISTERED_POSE2D_MODEL.register_module(ModelName.ResnetPose)
class ResnetPose(BasePoseModel):

    def __init__(self, data_channel=3, keypoints_number=17):
        super().__init__(data_channel, keypoints_number)
        self.set_name(ModelName.ResnetPose)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.ResNet50

        self.loss_config = {"type": LossName.JointsMSELoss,
                            "input_size": "192,256",
                            "reduction": 4,
                            "points_count": keypoints_number}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        num_deconv_layers = 3
        num_deconv_filters = (256, 256, 256)
        input_channel = self.block_out_channels[-1]
        for i in range(num_deconv_layers):
            deconv = DeConvBNActivationBlock(in_channels=input_channel,
                                             out_channels=num_deconv_filters[i],
                                             kernel_size=4,
                                             stride=2,
                                             padding=1,
                                             output_padding=0,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
            self.add_block_list(deconv.get_name(), deconv, num_deconv_filters[i])
            input_channel = num_deconv_filters[i]

        final_conv = nn.Conv2d(input_channel, self.keypoints_number, 1,
                               stride=1, padding=0)
        self.add_block_list(LayerType.Convolutional, final_conv, self.keypoints_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
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
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output
