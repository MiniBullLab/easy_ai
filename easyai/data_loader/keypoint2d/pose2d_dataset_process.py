#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.data_structure import Rect2D
from easyai.data_loader.common.box2d_dataset_process import Box2dDataSetProcess
from easyai.data_loader.augment.box2d_augment import Box2dAugment


class Pose2dDataSetProcess(Box2dDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.crop_augment = Box2dAugment()

    def crop_image(self, src_image, box, ratio, is_random=False):
        if is_random:
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            expand_box = self.get_expand_box(src_size, box, ratio)
            offset_box = self.crop_augment.random_crop_box(expand_box)
            xmin = int(np.clip(expand_box.min_corner.x + offset_box[0], 0, src_size[0]))
            ymin = int(np.clip(expand_box.min_corner.y + offset_box[1], 0, src_size[1]))
            xmax = int(np.clip(expand_box.max_corner.x + offset_box[2], 0, src_size[0]))
            ymax = int(np.clip(expand_box.max_corner.y + offset_box[3], 0, src_size[1]))
            expand_box = Rect2D()
            expand_box.min_corner.x = xmin
            expand_box.min_corner.y = ymin
            expand_box.max_corner.x = xmax
            expand_box.max_corner.y = ymax
            image = self.get_roi_image(src_image, expand_box)
        else:
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            expand_box = self.get_expand_box(src_size, box, ratio)
            image = self.get_roi_image(src_image, expand_box)
        return image, expand_box

    def crop_label(self, keypoint, expand_box):
        temp_points = keypoint.get_key_points()
        result = keypoint.copy()
        result.min_corner.x = keypoint.min_corner.x - expand_box.min_corner.x
        result.min_corner.y = keypoint.min_corner.y - expand_box.min_corner.y
        result.max_corner.x = keypoint.max_corner.x - expand_box.min_corner.x
        result.max_corner.y = keypoint.max_corner.y - expand_box.min_corner.y
        result.clear_key_points()
        for point in temp_points:
            if point.x < 0 or point.y < 0:
                result.add_key_points(point)
            else:
                point.x = point.x - expand_box.min_corner.x
                point.y = point.y - expand_box.min_corner.y
                result.add_key_points(point)
        return result

    def resize_dataset(self, src_image, image_size, keypoint, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_image, image_size)
        label = self.resize_label(keypoint, class_name, src_size, image_size)
        return image, label

    def normalize_label(self, keypoint):
        temp_points = keypoint.get_key_points()
        result = np.zeros((len(temp_points), 2), dtype=np.float)
        for index, point in enumerate(temp_points):
            result[index][0] = point.x
            result[index][1] = point.y
        return result

    def resize_label(self, keypoint, class_name, src_size, dst_size):
        result = None
        box = self.resize_box([keypoint], class_name, src_size, dst_size)
        if self.resize_type == 0:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            result = keypoint.copy()
            result.set_rect2d(box[0])
            result.clear_key_points()
            result.class_id = class_name.index(keypoint.name)
            temp_points = keypoint.get_key_points()
            for point in temp_points:
                if point.x < 0 or point.y < 0:
                    result.add_key_points(point)
                else:
                    point.x = ratio_w * point.x
                    point.y = ratio_h * point.y
                    result.add_key_points(point)
        elif self.resize_type == 1:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            result = keypoint.copy()
            result.set_rect2d(box[0])
            result.clear_key_points()
            result.class_id = class_name.index(keypoint.name)
            temp_points = keypoint.get_key_points()
            for point in temp_points:
                if point.x < 0 or point.y < 0:
                    result.add_key_points(point)
                else:
                    point.x = ratio * point.x + pad_size[0] // 2
                    point.y = ratio * point.y + pad_size[1] // 2
                    result.add_key_points(point)
        return result

