#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
from easyai.visualization.utility.color_define import ColorDefine


class ImageDrawing():

    def __init__(self):
        pass

    def draw_image(self, widnow_name, image, scale=1.0):
        cv2.namedWindow(widnow_name, 0)
        cv2.resizeWindow(widnow_name, int(image.shape[1] * scale), int(image.shape[0] * scale))
        cv2.imshow(widnow_name, image)

    def draw_detect_objects(self, src_image, result_objects):
        for object in result_objects:
            point1 = (int(object.min_corner.x), int(object.min_corner.y))
            point2 = (int(object.max_corner.x), int(object.max_corner.y))
            index = object.classIndex
            cv2.rectangle(src_image, point1, point2, ColorDefine.colors[index], 2)

    def draw_segment_result(self, src_image, result, class_list):
        r = result.copy()
        g = result.copy()
        b = result.copy()
        for index, value_data in enumerate(class_list):
            value = value_data[1]
            color_list = [int(x) for x in value.split(',') if x.strip()]
            if len(color_list) == 1:
                r[result == index] = color_list[0]
                g[result == index] = color_list[0]
                b[result == index] = color_list[0]
            elif len(color_list) == 3:
                r[result == index] = color_list[0]
                g[result == index] = color_list[1]
                b[result == index] = color_list[2]
            else:
                r[result == index] = 0
                g[result == index] = 0
                b[result == index] = 0

        rgb = np.zeros((result.shape[0], result.shape[1], 3))

        rgb[:, :, 0] = (r * 1.0 + src_image[:, :, 2] * 0) / 255.0
        rgb[:, :, 1] = (g * 1.0 + src_image[:, :, 1] * 0) / 255.0
        rgb[:, :, 2] = (b * 1.0 + src_image[:, :, 0] * 0) / 255.0

        return rgb

    def draw_keypoint2d_result(self, src_image, result, skeleton):
        for result_object in result:
            key_points = result_object.get_key_points()
            for edge in skeleton:
                point1 = key_points[edge[0]]
                point2 = key_points[edge[1]]
                x1 = int(point1.x)
                y1 = int(point1.y)
                x2 = int(point2.x)
                y2 = int(point2.y)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for point in key_points:
                cv2.circle(src_image, (int(point.x), int(point.y)), 5, (0, 255, 225), 2)

    def draw_det_keypoint2d_result(self, src_image, result, skeleton):
        for result_object in result:
            point1 = (int(result_object.min_corner.x), int(result_object.min_corner.y))
            point2 = (int(result_object.max_corner.x), int(result_object.max_corner.y))
            cv2.rectangle(src_image, point1, point2, (255, 255, 255), 2)
            key_points = result_object.get_key_points()
            for edge in skeleton:
                point1 = key_points[edge[0]]
                point2 = key_points[edge[1]]
                x1 = int(point1.x)
                y1 = int(point1.y)
                x2 = int(point2.x)
                y2 = int(point2.y)
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                cv2.line(src_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for point in key_points:
                cv2.circle(src_image, (int(point.x), int(point.y)), 5, (0, 255, 225), 2)

    def draw_polygon2d_result(self, src_image, result):
        for result_object in result:
            polygon = result_object.get_polygon()
            point_list = []
            for point in polygon:
                x = int(point.x)
                y = int(point.y)
                point_list.append([x, y])
            cv2.polylines(src_image, np.array([point_list], np.int32),
                          True, (0, 0, 225), 2)

