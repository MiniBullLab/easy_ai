#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import math
import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.image_data_augment import ImageDataAugment


class Pose2dDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        box = Rect2D()
        box.clear_key_points()
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        points = label.get_key_points()
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, degree = self.image_augment.augment_affine(image)
            points = self.label_affine(points, matrix, image_size)
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            points = self.label_lr_flip(points, is_lr, image_size)
        for point in points:
            box.add_key_points(point)
        return image, box

    def label_affine(self, points, matrix, image_size):
        for point in points:
            new_pt = np.array([point.x, point.y, 1.]).T
            new_pt = np.dot(matrix, new_pt)
            point.x = max(new_pt[0], 0)
            point.x = min(new_pt[0], image_size[0] - 1)
            point.y = max(new_pt[1], 0)
            point.y = min(new_pt[1], image_size[1] - 1)
        return points

    def label_lr_flip(self, points, is_lr, image_size):
        # left-right flip
        if is_lr:
            for point in points:
                point.x = image_size[0] - points.x

            for pair in self.flip_pairs:
                points[pair[0]], points[pair[1]] = \
                    points[pair[1]], points[pair[0]]
        return points

