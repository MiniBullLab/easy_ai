#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.helper.data_structure import DetectionObject
from easyai.loss.utility.box2d_process import box2d_xywh2xyxy
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.YoloPostProcess)
class YoloPostProcess(BasePostProcess):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.min_width = 2  # (pixels)
        self.min_height = 2  # (pixels)
        self.use_new_confidence = False

    def __call__(self, prediction):
        result = []
        class_confidence, class_index = prediction[:, 5:].max(1)
        if self.use_new_confidence:
            object_confidence = prediction[:, 4]
            object_confidence *= class_confidence
            temp1_indexs = object_confidence > self.threshold
        else:
            temp1_indexs = prediction[:, 4] > self.threshold
        temp2_indexs = (prediction[:, 2:4] > self.min_width).all(1)
        temp3_indexs = torch.isfinite(prediction).all(1)
        index_list = temp1_indexs & temp2_indexs & temp3_indexs
        prediction = prediction[index_list]
        class_confidence = class_confidence[index_list]
        class_index = class_index[index_list]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[:, :4] = box2d_xywh2xyxy(prediction[:, :4])
        for index, value in enumerate(prediction):
            temp_object = DetectionObject()
            temp_object.min_corner.x = value[0]
            temp_object.min_corner.y = value[1]
            temp_object.max_corner.x = value[2]
            temp_object.max_corner.y = value[3]
            temp_object.objectConfidence = value[4]
            temp_object.classConfidence = class_confidence[index]
            temp_object.classIndex = class_index[index]
            result.append(temp_object)
        return result
