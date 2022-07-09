#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class Box2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def resize_box(self, boxes, class_name, src_size, dst_size):
        labels = []
        if self.resize_type == 0:
            for box in boxes:
                rect = box.copy()
                rect.class_id = class_name.index(box.name)
                rect.min_corner.x = box.min_corner.x
                rect.min_corner.y = box.min_corner.y
                rect.max_corner.x = box.max_corner.x
                rect.max_corner.y = box.max_corner.y
                labels.append(rect)
        elif self.resize_type == 1:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            for box in boxes:
                rect = box.copy()
                rect.class_id = class_name.index(box.name)
                rect.min_corner.x = ratio_w * box.min_corner.x
                rect.min_corner.y = ratio_h * box.min_corner.y
                rect.max_corner.x = ratio_w * box.max_corner.x
                rect.max_corner.y = ratio_h * box.max_corner.y
                labels.append(rect)
        elif self.resize_type == 2:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            for box in boxes:
                rect = box.copy()
                rect.class_id = class_name.index(box.name)
                rect.min_corner.x = ratio * box.min_corner.x + pad_size[0] // 2
                rect.min_corner.y = ratio * box.min_corner.y + pad_size[1] // 2
                rect.max_corner.x = ratio * box.max_corner.x + pad_size[0] // 2
                rect.max_corner.y = ratio * box.max_corner.y + pad_size[1] // 2
                labels.append(rect)
        return labels

    def get_expand_box(self, src_size, box, ratio):
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
        rect2d = box.copy()
        rect2d.name = box.name
        rect2d.min_corner.x = new_left
        rect2d.min_corner.y = new_top
        rect2d.max_corner.x = new_right
        rect2d.max_corner.y = new_bottom
        return rect2d
