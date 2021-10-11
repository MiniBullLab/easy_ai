#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.keypoint2d.pose2d_dataset_process import Pose2dDataSetProcess


class LandmarkDataSetProcess(Pose2dDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def normalize_label(self, keypoint):
        rect = keypoint.get_rect2d()
        flags = keypoint.get_key_points_flag()
        temp_points = keypoint.get_key_points()
        points_result = np.zeros((len(temp_points), 2), dtype=np.float)
        for index, point in enumerate(temp_points):
            points_result[index][0] = point.x
            points_result[index][1] = point.y
        x, y = rect.center()
        width = rect.width()
        height = rect.height()
        box_result = [x, y, width, height]
        for flag in flags:
            box_result.append(flag)
        return points_result, np.array(box_result, dtype=np.float)
