#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.model.det2d.yolov3_det2d import YoloV3Det2d
from easyai.model.pose2d.resnet_pose import ResnetPose
from easyai.model.utility.base_pose_model import *
from easyai.model.utility.registry import REGISTERED_MULTI_MODEL


@REGISTERED_MULTI_MODEL.register_module(ModelName.DetPoes2dModel)
class DetPoes2dModel(BasePoseModel):

    def __init__(self, data_channel=3, keypoints_number=17):
        super().__init__(data_channel, keypoints_number)
        self.set_name(ModelName.DetPoes2dModel)

        self.det_model = None
        self.pose_model =None

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.det_model = YoloV3Det2d(data_channel=self.data_channel,
                                     class_number=1)

        self.pose_model = ResnetPose(data_channel=self.data_channel,
                                     keypoints_number=self.keypoints_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.clear_loss()

    def forward(self, x, net_type):
        output = []
        if net_type == 0:
            output = self.det_model(x)
        elif net_type == 1:
            output = self.pose_model(x)
        return output
