#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import DetectionKeyPoint
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.LandmarkPostProcess)
class LandmarkPostProcess(BasePostProcess):

    def __init__(self, points_count, threshold):
        super().__init__()
        self.points_count = points_count
        self.threshold = threshold

    def __call__(self, prediction):
        result = DetectionKeyPoint()
        coords = prediction[0]
        coords.view(self.points_count, 2)
        conf = prediction[1]
        valid_point = conf > self.threshold
        for index, valid in enumerate(valid_point):
            point = Point2d(-1, -1)
            if valid:
                point.x = int(coords[index][0])
                point.y = int(coords[index][1])
            result.add_key_points(point)
        return result
