#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import math
import numpy as np
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import Polygon2dObject
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess


class Polygon2dPostProcess():

    def __init__(self, image_size, post_process_config):
        self.image_size = image_size
        self.dataset_process = ImageDataSetProcess()

    def post_process(self, prediction, src_size, threshold=0.0):
        if prediction is None:
            return None
        detection_objects = self.get_polygon_result(prediction, threshold)
        result = self.resize_polygon_object(src_size, self.image_size, detection_objects)
        return result

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
