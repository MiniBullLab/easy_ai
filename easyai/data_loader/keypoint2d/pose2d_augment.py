#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.augment.image_data_augment import ImageDataAugment
from easyai.data_loader.augment.box2d_augment import Box2dAugment


class Pose2dDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.is_blur = False
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.image_augment = ImageDataAugment()
        self.box_augment = Box2dAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        keypoint = label.copy()
        keypoint.clear_key_points()
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        box = label.get_rect2d()
        points = label.get_key_points()
        if self.is_blur:
            image = self.image_augment.gaussian_blur(image_rgb)
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            copy_image = image[:]
            image, matrix, degree = self.image_augment.augment_affine(image)
            temp_box = self.box_augment.box_affine(box, matrix, degree, image_size)
            if temp_box is not None:
                box = temp_box
                points = self.label_affine(points, matrix, image_size)
            else:
                image = copy_image
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            if is_lr:
                box = self.box_augment.box_lr_flip(box, image_size)
            points = self.label_lr_flip(points, is_lr, image_size)
        for point in points:
            keypoint.add_key_points(point)
        keypoint.set_rect2d(box)
        return image, keypoint

    def label_affine(self, points, matrix, image_size):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            new_pt = np.array([point.x, point.y, 1.]).T
            new_pt = np.dot(matrix, new_pt)
            if (new_pt[0] < 0) or (new_pt[0] > image_size[0] - 1) \
                or (new_pt[1] < 0) or (new_pt[1] > image_size[1] - 1):
                point.x = -1
                point.y = -1
            else:
                point.x = new_pt[0]
                point.y = new_pt[1]
        return points

    def label_lr_flip(self, points, is_lr, image_size):
        # left-right flip
        if is_lr:
            for point in points:
                if point.x < 0 or point.y < 0:
                    continue
                point.x = image_size[0] - point.x - 1

            for pair in self.flip_pairs:
                points[pair[0]], points[pair[1]] = \
                    points[pair[1]], points[pair[0]]
        return points

