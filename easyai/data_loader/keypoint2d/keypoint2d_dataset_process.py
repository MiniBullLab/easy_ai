#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import KeyPoint2D
from easyai.data_loader.utility.box2d_dataset_process import Box2dDataSetProcess


class KeyPoint2dDataSetProcess(Box2dDataSetProcess):

    def __init__(self, points_count,
                 resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.points_count = points_count

    def resize_dataset(self, src_image, image_size, keypoints, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_size, image_size)
        labels = self.resize_labels(keypoints, class_name, src_size, image_size)
        return image, labels

    def normalize_labels(self, labels, image_size):
        result = np.zeros((len(labels), self.points_count * 2), dtype=np.float32)
        for index, keypoint in enumerate(labels):
            temp_data = []
            class_id = keypoint.class_id
            temp_data.append(class_id)
            temp_points = keypoint.get_key_points()
            for point in temp_points:
                x, y, = point.x, point.y
                x /= image_size[0]
                y /= image_size[1]
                temp_data.append(x)
                temp_data.append(y)
            result[index, :] = np.array(temp_data)
        return result

    def resize_labels(self, keypoints, class_name, src_size, dst_size):
        labels = []
        if self.resize_type == 0:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            for box_data in keypoints:
                if box_data.name in class_name:
                    result = KeyPoint2D()
                    result.class_id = class_name.index(box_data.name)
                    temp_points = box_data.get_key_points()
                    for point in temp_points:
                        point.x = ratio_w * point.x
                        point.y = ratio_h * point.y
                        result.add_key_points(point)
                    labels.append(result)
        elif self.resize_type == 1:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            for box_data in keypoints:
                if box_data.name in class_name:
                    result = KeyPoint2D()
                    result.class_id = class_name.index(box_data.name)
                    temp_points = result.get_key_points()
                    for point in temp_points:
                        point.x = ratio * point.x + pad_size[0] // 2
                        point.y = ratio * point.y + pad_size[0] // 2
                        result.add_key_points(point)
                    labels.append(result)
        return labels

    def change_outside_labels(self, labels):
        # reject warped points outside of image (0.999 for the image boundary)
        for i, label in enumerate(labels):
            for index in range(1, self.points_count+1):
                if label[index] >= float(1):
                    label[index] = 0.999
        return labels
