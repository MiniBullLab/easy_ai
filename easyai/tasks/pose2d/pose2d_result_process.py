#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import torch
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class Pose2dResultProcess():

    def __init__(self, points_count):
        self.use_new_confidence = False
        self.points_count = points_count
        self.dataset_process = ImageDataSetProcess()

    def postprocess(self, prediction, threshold=0.0):
        result_objects = None
        return result_objects

    def get_pose_result(self, prediction, conf_thresh, flag=0):
        pass
