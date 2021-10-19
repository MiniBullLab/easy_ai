#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import Rect2D
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess
from easyai.data_loader.common.polygon2d_process import Polygon2dProcess
from easyai.utility.logger import EasyLogger


class Polygon2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.polygon_process = Polygon2dProcess()

    def get_rotate_crop_image(self, src_image, polygon, expand_ratio):
        points = self.get_four_points(polygon)
        img_crop_width = int(
            max(np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        if img_crop_height * 1.0 / img_crop_width >= 1.5:
            new_width = int(img_crop_width * expand_ratio[1])
            new_height = int(img_crop_height * expand_ratio[0])
        else:
            new_width = int(img_crop_width * expand_ratio[0])
            new_height = int(img_crop_height * expand_ratio[1])
        M[0, 2] += (new_width - img_crop_width) / 2
        M[1, 2] += (new_height - img_crop_height) / 2
        dst_img = cv2.warpPerspective(src_image,
                                      M, (new_width, new_height),
                                      borderMode=cv2.BORDER_REPLICATE,
                                      flags=cv2.INTER_CUBIC,
                                      borderValue=self.pad_color)
        return dst_img

    def get_four_points(self, polygon):
        assert len(polygon) >= 4, EasyLogger.error(polygon)
        temp_points = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
        if len(polygon) > 4:
            # x_min = temp_points[:, 0].min()
            # x_max = temp_points[:, 0].max()
            # y_min = temp_points[:, 1].min()
            # y_max = temp_points[:, 1].max()
            # box = Rect2D(x_min, y_min, x_max, y_max)
            # dst_img = self.get_roi_image(src_image, box)
            rotated_box = cv2.minAreaRect(temp_points)
            temp_points = cv2.boxPoints(rotated_box)
        points = self.polygon_process.clockwise_coordinate_transformation(temp_points)
        points = self.polygon_process.original_coordinate_transformation(points)
        return points

    def rotation90_image(self, image, ratio=2.0):
        """
        anticlockwise rotate 90
        """
        dst_img = image[:]
        dst_img_height, dst_img_width = image.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= ratio:
            dst_img = np.rot90(image)
        return dst_img

    def polygon_area(self, polygon):
        temp_points = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
        return cv2.contourArea(temp_points)
        # edge = 0
        # for i in range(polygon.shape[0]):
        #     next_index = (i + 1) % polygon.shape[0]
        #     edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])
        #
        # return edge / 2.

    def resize_polygon(self, polygon, src_size, dst_size):
        result = []
        if self.resize_type == 0:
            result = polygon[:]
        elif self.resize_type == 1:
            ratio_w = float(dst_size[0]) / src_size[0]
            ratio_h = float(dst_size[1]) / src_size[1]
            for point in polygon:
                x = ratio_w * point.x
                y = ratio_h * point.y
                result.append(Point2d(x, y))
        elif self.resize_type == 2:
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            for point in polygon:
                x = ratio * point.x + pad_size[0] // 2
                y = ratio * point.y + pad_size[1] // 2
                result.append(Point2d(x, y))
        elif self.resize_type == 4:
            resize_w, resize_h = self.dataset_process.get_short_size(src_size, dst_size)
            ratio_w = float(resize_w) / src_size[0]
            ratio_h = float(resize_h) / src_size[1]
            for point in polygon:
                x = ratio_w * point.x
                y = ratio_h * point.y
                result.append(Point2d(x, y))
        return result


