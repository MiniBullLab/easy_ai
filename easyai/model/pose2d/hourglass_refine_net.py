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


@REGISTERED_POSE2D_MODEL.register_module(ModelName.HourglassRefineNet)
class HourglassRefineNet(BasePoseModel):

    def __init__(self, data_channel=3, points_count=68):
        super().__init__(data_channel, points_count)
        self.set_name(ModelName.HourglassRefineNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
