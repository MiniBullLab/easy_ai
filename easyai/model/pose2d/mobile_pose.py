#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType, LayerType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.utility_layer import MeanLayer
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.model_block.base_block.utility.upsample_layer import DenseUpsamplingConvBlock
from easyai.model.utility.base_pose_model import *
from easyai.model.utility.registry import REGISTERED_POSE2D_MODEL


@REGISTERED_POSE2D_MODEL.register_module(ModelName.MobilePose)
class MobilePose(BasePoseModel):

    def __init__(self, data_channel=3, keypoints_number=16):
        super().__init__(data_channel, keypoints_number)
        self.set_name(ModelName.MobilePose)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU6

        self.model_args['type'] = BackboneName.MobileNetV2_1_0

        self.loss_config = {"type": LossName.DSNTLoss}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = MeanLayer(3)
        self.add_block_list(layer1.get_name(), layer1, self.block_out_channels[-1])

        layer2 = MeanLayer(2)
        self.add_block_list(layer2.get_name(), layer2, self.block_out_channels[-1])

        conv1 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                      out_channels=1280,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 1280)

        conv_compress = nn.Conv2d(1280, 256, 1, 1, 0, bias=False)
        self.add_block_list(LayerType.Convolutional, conv_compress, 256)

        duc1 = DenseUpsamplingConvBlock(256, 512, upscale_factor=2)
        self.add_block_list(duc1.get_name(), duc1, duc1.get_output_channel())
        duc2 = DenseUpsamplingConvBlock(duc1.get_output_channel(), 256, upscale_factor=2)
        self.add_block_list(duc2.get_name(), duc2, duc2.get_output_channel())
        duc3 = DenseUpsamplingConvBlock(duc2.get_output_channel(), 128, upscale_factor=2)
        self.add_block_list(duc3.get_name(), duc3, duc3.get_output_channel())

        hm_conv = nn.Conv2d(duc3.get_output_channel(), self.keypoints_number,
                            kernel_size=1, bias=False)
        self.add_block_list(LayerType.Convolutional, hm_conv, self.keypoints_number)

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
