#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.augment.image_data_augment import ImageDataAugment


class RecTextDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        ocr_object = label.copy()
        ocr_object.clear_polygon()
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        points = label.get_polygon()
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, degree = self.image_augment.augment_affine(image)
            points = self.label_affine(points, matrix, image_size)
        for point in points:
            ocr_object.add_point(point)
        return image, ocr_object

    def label_affine(self, points, matrix, image_size):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            new_pt = np.array([point.x, point.y, 1.]).T
            new_pt = np.dot(matrix, new_pt)
            if new_pt[0] < 0:
                point.x = 0
            elif new_pt[0] > image_size[0] - 1:
                point.x = image_size[0] - 1
            else:
                point.x = new_pt[0]
            if new_pt[1] < 0:
                point.y = 0
            elif new_pt[1] > image_size[1] - 1:
                point.y = image_size[1] - 1
            else:
                point.y = new_pt[1]
        return points
