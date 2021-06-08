#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import Polygon2dObject
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.registry import build_from_cfg


class Polygon2dResultProcess(BasePostProcess):

    def __init__(self, image_size, post_process_args):
        super().__init__()
        self.image_size = image_size
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction, src_size):
        if prediction is None:
            return None
        detection_objects = self.process_func(prediction, src_size)
        # result = self.resize_polygon_object(src_size, self.image_size, detection_objects)
        return detection_objects

    def resize_polygon_object(self, src_size, image_size,
                              detection_objects):
        result = []
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for obj in detection_objects:
            temp_object = Polygon2dObject()
            temp_object.clear_polygon()
            temp_object.class_id = int(obj.classIndex)
            temp_object.object_confidence = obj.object_confidence
            for value in obj.get_polygon():
                if value.x >= 0 and value.y >= 0:
                    x = int((value.x - pad[0] // 2) / ratio)
                    y = int((value.y - pad[1] // 2) / ratio)
                    point = Point2d(x, y)
                else:
                    point = value
                temp_object.add_point(point)
            result.append(temp_object)
        return result

    def build_post_process(self, post_process_args):
        func_name = post_process_args.strip()
        result_func = None
        if REGISTERED_POST_PROCESS.has_class(func_name):
            result_func = build_from_cfg(post_process_args, REGISTERED_POST_PROCESS)
        else:
            print("%s post process not exits" % func_name)
        return result_func
