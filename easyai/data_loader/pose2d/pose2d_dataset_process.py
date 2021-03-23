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
        new_left = int(np.clip(xmin - ratio * width, 0, src_size[0]))
        new_right = int(np.clip(xmax + ratio * width, 0, src_size[0]))
        new_top = int(np.clip(ymin - ratio * height, 0, src_size[1]))
        new_bottom = int(np.clip(ymax + ratio * height, 0, src_size[1]))
        if len(src_image.shape) == 3:
            image = src_image[new_top:new_bottom, new_left:new_right, :]
        elif len(src_image.shape) == 2:
            image = src_image[new_top:new_bottom, new_left:new_right]
        else:
            image = None
        points = box.get_key_points()
        rect2d = Rect2D()
        rect2d.name = box.name
        rect2d.min_corner.x = new_left
        rect2d.min_corner.y = new_top
        rect2d.max_corner.x = new_right
        rect2d.max_corner.y = new_bottom
        rect2d.clear_key_points()
        for point in points:
            if point.x < 0 or point.y < 0:
                rect2d.add_key_points(point)
            else:
                point.x = point.x - new_left
                point.y = point.y - new_top
                rect2d.add_key_points(point)
        return image, rect2d

    def resize_dataset(self, src_image, image_size, box, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_image, image_size)
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
            result = Rect2D()
            result.clear_key_points()
            result.class_id = class_name.index(box.name)
            key_points = box.get_key_points()
            for point in key_points:
                if point.x < 0 or point.y < 0:
                    result.add_key_points(point)
                else:
                    point.x = ratio_w * point.x
                    point.y = ratio_h * point.y
                    result.add_key_points(point)
        elif self.resize_type == 1:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            result = Rect2D()
            result.clear_key_points()
            result.class_id = class_name.index(box.name)
            key_points = box.get_key_points()
            for point in key_points:
                if point.x < 0 or point.y < 0:
                    result.add_key_points(point)
                else:
                    point.x = ratio * point.x + pad_size[0] // 2
                    point.y = ratio * point.y + pad_size[1] // 2
                    result.add_key_points(point)
        return result

