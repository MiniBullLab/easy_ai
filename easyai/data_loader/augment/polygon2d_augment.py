#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np


class Polygon2dAugment():

    def __init__(self):
        pass

    def polygon_affine(self, points, matrix):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            new_pt = np.array([point.x, point.y, 1.]).T
            new_pt = np.dot(matrix, new_pt)
            point.x = new_pt[0]
            point.y = new_pt[1]
        return points

    def polygon_lr_flip(self, points, image_size):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            point.x = image_size[0] - point.x
        return points

    def polygon_up_flip(self, points, image_size):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            point.y = image_size[1] - point.y
        return points

