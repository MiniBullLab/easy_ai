#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.model_block.base_block.common.utility_block import FcBNActivationBlock
from easyai.model_block.base_block.common.utility_layer import NormalizeLayer
from easyai.model_block.base_block.common.utility_layer import ActivationLayer

from easy_pc.name_manager.pc_model_name import PCModelName
from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.model.utility.base_pc_classify_model import *
from easy_pc.name_manager.pc_backbone_name import PCBackboneName
from easy_pc.model.utility.pc_model_registry import REGISTERED_PC_CLS_MODEL


@REGISTERED_PC_CLS_MODEL.register_module(PCModelName.PointNetCls)
class PointNetCls(BasePCClassifyModel):

    def __init__(self, data_channel=3, class_number=40):
        super().__init__(data_channel, class_number)
        self.set_name(PCModelName.PointNetCls)

        self.feature_transform = False
        self.bn_name = NormalizationType.BatchNormalize1d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = PCBackboneName.PointNet
        self.model_args['feature_transform'] = self.feature_transform
        self.model_args['bn_name'] = self.bn_name
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
        fc2 = nn.Linear(input_channel, 256)
        self.add_block_list(LayerType.FcLinear, fc2, 256)

        input_channel = 256
        dropout = nn.Dropout(p=0.3)
        self.add_block_list(LayerType.Dropout, dropout, input_channel)

        normalize = NormalizeLayer(self.bn_name, input_channel)
        self.add_block_list(normalize.get_name(), normalize, input_channel)

        activate = ActivationLayer(self.activation_name, inplace=False)
        self.add_block_list(activate.get_name(), activate, input_channel)

        fc3 = nn.Linear(input_channel, self.class_number)
        self.add_block_list(LayerType.FcLinear, fc3, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': PCLossName.PointNetClsLoss,
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
            elif PCLossName.PointNetClsLoss in key:
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        output.extend(trans)
        return output
