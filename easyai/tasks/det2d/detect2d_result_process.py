#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.tasks.utility.task_result_process import TaskPostProcess
from easyai.helper.data_structure import DetectionObject
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression
from easyai.utility.logger import EasyLogger


class Detect2dResultProcess(TaskPostProcess):

    def __init__(self, image_size, detect2d_class,
                 post_process_args):
        super().__init__()
        self.post_process_args = post_process_args
        self.nms_threshold = post_process_args.pop('nms_threshold')
        self.image_size = image_size
        self.detect2d_class = detect2d_class
        self.nms_process = FastNonMaxSuppression()
        self.process_func = self.build_post_process(post_process_args)
        EasyLogger.debug("det2d class name:{}".format(self.detect2d_class))

    def post_process(self, prediction, src_size):
        if prediction is None:
            return None
        result = self.process_func(prediction)
        detection_objects = self.nms_process.multi_class_nms(result, self.nms_threshold)
        detection_objects = self.resize_detection_objects(src_size,
                                                          self.image_size,
                                                          detection_objects,
                                                          self.detect2d_class)
        return detection_objects

    def resize_detection_objects(self, src_size, image_size,
                                 detection_objects, class_name):
        result = []
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for obj in detection_objects:
            temp_object = DetectionObject()
            x1 = (obj.min_corner.x - pad[0] // 2) / ratio
            y1 = (obj.min_corner.y - pad[1] // 2) / ratio
            x2 = (obj.max_corner.x - pad[0] // 2) / ratio
            y2 = (obj.max_corner.y - pad[1] // 2) / ratio
            temp_object.min_corner.x = x1
            temp_object.min_corner.y = y1
            temp_object.max_corner.x = x2
            temp_object.max_corner.y = y2
            temp_object.classIndex = int(obj.classIndex)
            temp_object.objectConfidence = obj.objectConfidence
            temp_object.classConfidence = obj.classConfidence
            temp_object.name = class_name[temp_object.classIndex]
            result.append(temp_object)
        return result

