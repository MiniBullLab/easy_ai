#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import PointCloudBackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.utility_block import FcBNActivationBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_PC_CLS_MODEL


@REGISTERED_PC_CLS_MODEL.register_module(ModelName.PointNet2Cls)
class PointNet2Cls(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=40):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.PointNetCls)

        self.bn_name = NormalizationType.BatchNormalize1d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = PointCloudBackboneName.PointNet
        self.model_args['activation_name'] = self.activation_name

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        input_channel = self.block_out_channels[-1]
        fc1 = FcBNActivationBlock(input_channel, 512,
                                  bnName=self.bn_name,
                                  activationName=self.activation_name)
        self.add_block_list(fc1.get_name(), fc1, 512)

        input_channel = 512
        dropout1 = nn.Dropout(p=0.4)
        self.add_block_list(LayerType.Dropout, dropout1, input_channel)

        fc2 = FcBNActivationBlock(input_channel, 256,
                                  bnName=self.bn_name,
                                  activationName=self.activation_name)
        self.add_block_list(fc2.get_name(), fc2, 256)

        input_channel = 256
        dropout2 = nn.Dropout(p=0.4)
        self.add_block_list(LayerType.Dropout, dropout2, input_channel)

        fc3 = nn.Linear(input_channel, self.class_number)
        self.add_block_list(LayerType.FcLinear, fc3, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.PointNetClsLoss,
                       'flag': False}
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
            elif LossName.PointNetClsLoss in key:
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
