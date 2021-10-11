#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import DetectionKeyPoint
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.MobilePostProcess)
class MobilePostProcess(BasePostProcess):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def __call__(self, prediction):
        result = DetectionKeyPoint()
        x = (prediction.reshape([-1, 2]) + np.array([1.0, 1.0])) / 2.0
        x = x * np.array(self.input_size)
        for value in x:
            point = Point2d(int(value[0]), int(value[1]))
            result.add_key_points(point)
        return result

