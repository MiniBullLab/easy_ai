#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.helper.data_structure import Point2d, DetectionObject
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.YoloKeypointPostProcess)
class YoloKeypointPostProcess(BasePostProcess):

    def __init__(self, points_count, threshold):
        super().__init__()
        self.points_count = points_count
        self.threshold = threshold
        self.use_new_confidence = False

    def __call__(self, prediction):
        result = []
        loc_count = self.points_count * 2
        class_confidence, class_index = prediction[:, loc_count + 1:].max(1)
        if self.use_new_confidence:
            object_confidence = prediction[:, loc_count]
            object_confidence *= class_confidence
            temp1_indexs = object_confidence > self.threshold
        else:
            temp1_indexs = prediction[:, loc_count] > self.threshold
        temp2_indexs = torch.isfinite(prediction).all(1)
        index_list = temp1_indexs & temp2_indexs
        prediction = prediction[index_list]
        class_confidence = class_confidence[index_list]
        class_index = class_index[index_list]
        # (x1, y1, x2, y2, ...)
        best_confidence = -1
        for index, value in enumerate(prediction):
            temp_object = DetectionObject()
            for index in range(0, self.points_count, 2):
                point = Point2d(value[index], value[index + 1])
                temp_object.add_key_points(point)
            temp_object.objectConfidence = value[4]
            temp_object.classConfidence = class_confidence[index]
            temp_object.classIndex = class_index[index]
            if best_confidence == -1:
                result.append(temp_object)
                best_confidence = temp_object.objectConfidence
            elif temp_object.objectConfidence > best_confidence:
                result[0] = temp_object
        return result
