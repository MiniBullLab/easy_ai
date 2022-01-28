#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import math
import random
import numpy as np


class Box2dAugment():

    def __init__(self):
        pass

    def random_crop_box(self, box, trans_ratio=0.1):
        translate_param1 = int(trans_ratio * abs(box.max_corner.x - box.min_corner.x))
        translate_param2 = int(trans_ratio * abs(box.max_corner.y - box.min_corner.y))
        translation_x1 = random.randint(-translate_param1, translate_param1)
        translation_x2 = random.randint(-translate_param1, translate_param1)
        translation_y1 = random.randint(-translate_param2, translate_param2)
        translation_y2 = random.randint(-translate_param2, translate_param2)
        return translation_x1, translation_y1, translation_x2, translation_y2

    def box_affine(self, box, matrix, degree, image_size):
        points = np.array(box.getVector())
        area0 = (points[2] - points[0]) * (points[3] - points[1])
        xy = np.ones((4, 3))
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)
        xy = np.squeeze((xy @ matrix.T)[:, :2].reshape(1, 8))

        # create new boxes
        x = xy[[0, 2, 4, 6]]
        y = xy[[1, 3, 5, 7]]
        xy = np.array([x.min(), y.min(), x.max(), y.max()])

        # apply angle-based reduction
        radians = degree * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[2] + xy[0]) / 2
        y = (xy[3] + xy[1]) / 2
        w = (xy[2] - xy[0]) * reduction
        h = (xy[3] - xy[1]) * reduction
        xy = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

        # reject warped points outside of image
        np.clip(xy, 0, image_size[0], out=xy)
        w = xy[2] - xy[0]
        h = xy[3] - xy[1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)
        if i:
            rect = box.copy()
            rect.min_corner.x = xy[0]
            rect.min_corner.y = xy[1]
            rect.max_corner.x = xy[2]
            rect.max_corner.y = xy[3]
            return rect
        else:
            return None

    def box_lr_flip(self, box, image_size):
        # left-right flip
        result = box.copy()
        x, y = box.center()
        width = box.width()
        x = image_size[0] - x
        result.min_corner.x = x - width / 2
        result.max_corner.x = x + width / 2
        return result

    def box_up_flip(self, box, image_size):
        # up-down flip
        result = box.copy()
        x, y = box.center()
        height = box.height()
        y = image_size[1] - y
        result.min_corner.y = y - height / 2
        result.max_corner.y = y + height / 2
        return result
