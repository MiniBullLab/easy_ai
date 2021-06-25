#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

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

    def original_coordinate_transformation(self, polygon):
        """
        调整坐标顺序为：
          x1,y1    x2,y2
          x4,y4    x3,y3
        :param polygon:
        :return:
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = polygon.astype(float).reshape(-1)
        # 判断x1和x3大小，x3调整为大的数
        if x1 > x3:
            x1, y1, x3, y3 = x3, y3, x1, y1
        # 判断x2和x4大小，x4调整为大的数
        if x2 > x4:
            x2, y2, x4, y4 = x4, y4, x2, y2
        # 判断y1和y2大小，y1调整为大的数
        if y2 > y1:
            x2, y2, x1, y1 = x1, y1, x2, y2
        # 判断y3和y4大小，y4调整为大的数
        if y3 > y4:
            x3, y3, x4, y4 = x4, y4, x3, y3
        return np.array([[x2, y2], [x3, y3], [x4, y4], [x1, y1]], dtype=np.float32)
