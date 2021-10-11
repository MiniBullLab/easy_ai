#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import DetectionKeyPoint
from easyai.tasks.utility.task_result_process import TaskPostProcess


class LandmarkResultProcess(TaskPostProcess):

    def __init__(self, points_count, image_size,
                 post_process_args):
        super().__init__()
        self.points_count = points_count
        self.image_size = image_size
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction, src_size):
        if prediction is None:
            return None
        object_landmark = self.process_func(prediction)
        result = self.resize_object_pose(src_size, self.image_size, object_landmark)
        return object_landmark, result

    def resize_object_pose(self, src_size, image_size,
                           object_pose):
        result = DetectionKeyPoint()
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for value in object_pose.get_key_points():
            if value.x != -1 and value.y != -1:
                x = int((value.x - pad[0] // 2) / ratio)
                y = int((value.y - pad[1] // 2) / ratio)
                point = Point2d(x, y)
            else:
                point = value
            result.add_key_points(point)
        return result
