#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import DetectionKeyPoint
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess


class LandmarkResultProcess():

    def __init__(self, post_prcoess_type, points_count, image_size):
        self.post_prcoess_type = post_prcoess_type
        self.points_count = points_count
        self.image_size = image_size
        self.dataset_process = ImageDataSetProcess()

    def postprocess(self, prediction, src_size, threshold=0.0):
        if prediction is None:
            return None
        object_landmark = self.get_landmark_result(prediction, threshold)
        result = self.resize_object_pose(src_size, self.image_size, object_landmark)
        return result

    def get_landmark_result(self, prediction, conf_thresh):
        result = None
        if self.post_prcoess_type == 0:
            result = self.get_face_landmark_result(prediction, conf_thresh)
        return result

    def get_face_landmark_result(self, prediction, conf_thresh):
        result = DetectionKeyPoint()
        coords = prediction[0]
        coords.view(self.points_count, 2)
        conf = prediction[1]
        valid_point = conf > conf_thresh
        for index, valid in enumerate(valid_point):
            point = Point2d(-1, -1)
            if valid:
                point.x = int(coords[index][0])
                point.y = int(coords[index][1])
            result.add_key_points(point)
        return result

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
