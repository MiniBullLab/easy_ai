#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np


class Polygon2dProcess():

    def __init__(self):
        pass

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
