#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.helper.data_structure import DetectionObject
from easyai.tasks.utility.task_result_process import TaskPostProcess


class KeyPoint2dResultProcess(TaskPostProcess):

    def __init__(self, image_size, points_count, points_class,
                 post_process_args):
        super().__init__()
        self.image_size = image_size
        self.points_class = points_class
        self.points_count = points_count
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction, src_size):
        if prediction is None:
            return None
        result = self.process_func(prediction)
        result_objects = self.resize_keypoints_objects(src_size,
                                                       self.image_size,
                                                       result,
                                                       self.points_class)
        return result, result_objects

    def resize_keypoints_objects(self, src_size, image_size,
                                 result_objects, class_name):
        result = []
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for obj in result_objects:
            temp_object = DetectionObject()
            for index in range(0, self.points_count, 2):
                temp_object.key_points[index].x = (temp_object.key_points[index].x -
                                                   pad[0] // 2) / ratio
                temp_object.key_points[index].y = (temp_object.key_points[index].y -
                                                   pad[1] // 2) / ratio
            temp_object.classIndex = int(obj.classIndex)
            temp_object.objectConfidence = obj.objectConfidence
            temp_object.classConfidence = obj.classConfidence
            temp_object.name = class_name[temp_object.classIndex]
            result.append(temp_object)
        return result

