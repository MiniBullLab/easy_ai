#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import PointCloudBackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock1d
from easyai.model_block.base_block.pc_cls.pointnet_block import PointNetRouteBlock
from easyai.model_block.base_block.pc_cls.pointnet_block import PointNetBlockName
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_PC_SEG_MODEL


@REGISTERED_PC_SEG_MODEL.register_module(ModelName.PointNetSeg)
class PointNetSeg(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=13):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.PointNetSeg)

        self.feature_transform = True
        self.bn_name = NormalizationType.BatchNormalize1d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = PointCloudBackboneName.PointNet
        self.model_args['feature_transform'] = self.feature_transform
        self.model_args['bn_name'] = self.bn_name
        self.model_args['activation_name'] = self.activation_name

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        route_block = PointNetRouteBlock(1)
        self.add_block_list(route_block.get_name(), route_block, 1024 + 64)

        input_channel = 1024 + 64
        conv1 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=512,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        input_channel = 512
        conv2 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=256,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 256)

        input_channel = 256
        conv3 = ConvBNActivationBlock1d(in_channels=input_channel,
                                        out_channels=128,
                                        kernel_size=1,
                                        bnName=self.bn_name,
                                        activationName=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 128)

        input_channel = 128
        conv4 = nn.Conv1d(input_channel, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional1d, conv4, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.PointNetSegLoss,
                       'flag': self.feature_transform}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        trans = []
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
                trans = block.get_transform_output()
            elif PointNetBlockName.PointNetRouteBlock in key:
                x = block(x, base_outputs)
            elif LossName.PointNetSegLoss in key:
                output.append(x)
            else:
                x = block(x)
            print(key, x.shape)
            layer_outputs.append(x)
        output.extend(trans)
        return output