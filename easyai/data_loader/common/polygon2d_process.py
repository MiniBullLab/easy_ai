#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import math
import operator
from functools import reduce
import numpy as np


class Polygon2dProcess():

    def __init__(self):
        pass

    def rotate_points(self, _points, _degree=0, _center=(0, 0)):
        """
        逆时针绕着一个点旋转点
        Args:
            _points:    需要旋转的点
            _degree:    角度
            _center:    中心点
        Returns:    旋转后的点
        """
        angle = np.deg2rad(_degree)
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        center = np.atleast_2d(_center)
        points = np.atleast_2d(_points)
        return np.squeeze((rotate_matrix @ (points.T - center.T) + center.T).T)

    def clockwise_coordinate_transformation(self, polygon):
        assert len(polygon) > 2
        coords = list(polygon)
        center = tuple(map(operator.truediv,
                           reduce(lambda x, y: map(operator.add, x, y), coords),
                           [len(coords)] * 2))
        coords = sorted(coords, key=lambda coord: \
                (-180 + math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))))
        return np.array(coords)
