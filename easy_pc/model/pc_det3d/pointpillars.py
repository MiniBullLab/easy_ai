#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import BlockType
from easyai.name_manager.loss_name import LossName

from easy_pc.name_manager.pc_block_name import PCEncoderName
from easy_pc.name_manager.pc_model_name import PCModelName
from easy_pc.name_manager.pc_backbone_name import PCBackboneName
from easy_pc.name_manager.pc_block_name import PCNeckType, PCHeadType
from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.model.utility.base_pc_det3d_model import *
from easy_pc.model_block.encoder.pillar_encoder import PillarFeatureNet
from easy_pc.model_block.encoder.pillar_scatter import PointPillarsScatter
from easy_pc.model_block.neck.second_fpn_neck import SecondFPNNeck
from easy_pc.model_block.head.det3d_box_head import Detection3dBoxHead
from easy_pc.model.utility.pc_model_registry import REGISTERED_PC_DET3D_MODEL


@REGISTERED_PC_DET3D_MODEL.register_module(PCModelName.PointPillars)
class PointPillars(BasePCDet3dModel):

    def __init__(self, data_channel=3, class_number=3):
        super().__init__(data_channel, class_number)
        self.set_name(PCModelName.PointPillars)

        self.act_name = ActivationType.ReLU

        self.model_args['data_channel'] = 64
        self.model_args['type'] = PCBackboneName.SecondNet
        self.model_args['bn_name'] = NormalizationType.BatchNormalize2d
        self.model_args['act_name'] = self.act_name

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        voxel_encoder = PillarFeatureNet(in_channels=self.data_channel,
                                         feat_channels=(64,),
                                         voxel_size=(0.16, 0.16, 4),
                                         point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1))
        self.add_block_list(voxel_encoder.get_name(), voxel_encoder, 64)

        middle_encoder = PointPillarsScatter(in_channels=64,
                                             output_shape=[496, 432])
        self.add_block_list(middle_encoder.get_name(), middle_encoder, 64)

        backbone = self.pc_backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        neck = SecondFPNNeck("3,9,15",
                             down_channels=[64, 128, 256],
                             out_channels=[128, 128, 128],
                             upsample_strides=[1, 2, 4])
        self.add_block_list(neck.get_name(), neck, 128 + 128 + 128)

        head = Detection3dBoxHead(input_channle=self.block_out_channels[-1],
                                  anchor_number=self.class_number * 2,
                                  class_number=self.class_number)
        self.add_block_list(head.get_name(), head, -1)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.EmptyLoss}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x_list):
        trans = []
        base_outputs = []
        layer_outputs = []
        output = []
        x, num_points, coors = x_list
        for key, block in self._modules.items():
            if PCEncoderName.PillarFeatureBlock in key:
                x = self.block(x, num_points, coors)
            elif PCEncoderName.PointPillarsScatter in key:
                batch_size = coors[-1, 0].item() + 1
                x = self.middle_encoder(x, coors, batch_size)
            elif BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif PCNeckType.SecondFPNNeck in key:
                x = block(layer_outputs, base_outputs)
            elif PCHeadType.Detection3dBoxHead in key:
                x = block(x)
            elif LossName.EmptyLoss in key:
                output.extend(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        output.extend(trans)
        return output
