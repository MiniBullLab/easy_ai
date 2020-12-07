#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class KeyPoint2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, points_count,
                 resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.points_count = points_count

    def resize_dataset(self, src_image, image_size, boxes, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_size, image_size)
        labels = self.resize_labels(boxes, class_name, src_size, image_size)
        return image, labels

    def normalize_labels(self, labels, image_size):
        result = np.zeros((len(labels), self.points_count * 2), dtype=np.float32)
        for index, rect2d in enumerate(labels):
            temp_data = []
            class_id = rect2d.class_id
            temp_data.append(class_id)
            key_points = rect2d.get_key_points()
            for point in key_points:
                x, y, = point.x, point.y
                x /= image_size[0]
                y /= image_size[1]
                temp_data.append(x)
                temp_data.append(y)
            result[index, :] = np.array(temp_data)
        return result

    def resize_labels(self, boxes, class_name, src_size, dst_size):
        labels = []
        if self.resize_type == 0:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            for box in boxes:
                if box.name in class_name:
                    rect2d = Rect2D()
                    rect2d.class_id = class_name.index(box.name)
                    key_points = box.get_key_points()
                    for point in key_points:
                        point.x = ratio_w * point.x
                        point.y = ratio_h * point.y
                        rect2d.add_key_points(point)
                    labels.append(rect2d)
        elif self.resize_type == 1:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            for box in boxes:
                if box.name in class_name:
                    rect2d = Rect2D()
                    rect2d.class_id = class_name.index(box.name)
                    key_points = box.get_key_points()
                    for point in key_points:
                        point.x = ratio * point.x + pad_size[0] // 2
                        point.y = ratio * point.y + pad_size[0] // 2
                        rect2d.add_key_points(point)
                    labels.append(rect2d)
        return labels

    def change_outside_labels(self, labels):
        # reject warped points outside of image (0.999 for the image boundary)
        for i, label in enumerate(labels):
            for index in range(1, self.points_count+1):
                if label[index] >= float(1):
                    label[index] = 0.999
        return labels
