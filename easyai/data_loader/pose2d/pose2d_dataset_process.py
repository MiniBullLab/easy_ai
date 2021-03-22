#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class Pose2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def expand_dataset(self, src_image, box, ratio):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        xmin = box.min_corner.x
        ymin = box.min_corner.y
        xmax = box.max_corner.x
        ymax = box.max_corner.y
        width = xmax - xmin
        height = ymax - ymin
        new_left = np.clip(xmin - ratio * width, 0, src_size[0])
        new_right = np.clip(xmax + ratio * width, 0, src_size[0])
        new_top = np.clip(ymin - ratio * height, 0, src_size[1])
        new_bottom = np.clip(ymax + ratio * height, 0, src_size[1])
        if len(src_image.shape) == 3:
            image = src_image[new_top:new_bottom, new_left:new_right, :]
        elif len(src_image.shape) == 2:
            image = src_image[new_top:new_bottom, new_left:new_right]
        else:
            image = None
        box.min_corner.x = new_left
        box.min_corner.y = new_top
        box.max_corner.x = new_right
        box.max_corner.y = new_bottom
        points = box.get_key_points()
        box.clear_key_points()
        for point in points:
            point.x = point.x - new_left
            point.y = point.y - new_top
            box.add_key_points(point)
        return image, box

    def resize_dataset(self, src_image, image_size, box, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_size, image_size)
        label = self.resize_label(box, class_name, src_size, image_size)
        return image, label

    def normalize_label(self, box):
        key_points = box.get_key_points()
        result = np.zeros((len(key_points), 2), dtype=np.float)
        for index, point in enumerate(key_points):
            result[index][0] = point.x
            result[index][1] = point.y
        return result

    def resize_label(self, box, class_name, src_size, dst_size):
        result = None
        if self.resize_type == 0:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            rect2d = Rect2D()
            rect2d.class_id = class_name.index(box.name)
            key_points = box.get_key_points()
            for point in key_points:
                point.x = ratio_w * point.x
                point.y = ratio_h * point.y
                rect2d.add_key_points(point)
            result = rect2d
        elif self.resize_type == 1:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            rect2d = Rect2D()
            rect2d.class_id = class_name.index(box.name)
            key_points = box.get_key_points()
            for point in key_points:
                point.x = ratio * point.x + pad_size[0] // 2
                point.y = ratio * point.y + pad_size[0] // 2
                rect2d.add_key_points(point)
            result = rect2d
        return result

