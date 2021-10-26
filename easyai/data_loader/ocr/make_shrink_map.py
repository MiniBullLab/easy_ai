#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

__all__ = ['MakeShrinkMap']


def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class MakeShrinkMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, min_text_size=8, shrink_ratio=0.4, shrink_type='pyclipper'):
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.number = 0

    def __call__(self, data: dict) -> dict:
        image = data['image']
        text_polys = data['text_polys']
        h, w = image.shape[-2:]
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if min(height, width) < self.min_text_size:
                # print("min size")
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    # print("shrinked.size==0")
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        # cv2.imwrite("img_map_%d.png" % self.number, gt * 255)
        # cv2.imwrite("img_mask_%d.png" % self.number, mask * 255)
        # self.number += 1
        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data


if __name__ == '__main__':
    polygon = np.array([[0, 0], [100, 10], [100, 100], [10, 90]])
    a = shrink_polygon_py(polygon, 0.4)
    print(a)
    print(shrink_polygon_py(a, 1 / 0.4))
    b = shrink_polygon_pyclipper(polygon, 0.4)
    print(b)
    poly = Polygon(b)
    distance = poly.area * 1.5 / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(b, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    bounding_box = cv2.minAreaRect(expanded)
    points = cv2.boxPoints(bounding_box)
    print(points)
