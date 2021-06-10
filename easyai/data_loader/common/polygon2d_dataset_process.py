#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
from easyai.helper.data_structure import Rect2D
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess
from easyai.data_loader.common.polygon2d_process import Polygon2dProcess


class Polygon2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.polygon_process = Polygon2dProcess()

    def get_rotate_crop_image(self, src_image, polygon):
        assert len(polygon) >= 4
        points = np.array([[p.x, p.y] for p in polygon], dtype=np.float32)
        if len(polygon) > 4:
            rect = Rect2D()
            rect.min_corner.x = points[:, 0].min()
            rect.min_corner.x = points[:, 0].max()
            rect.max_corner.y = points[:, 1].min()
            rect.max_corner.y = points[:, 1].max()
            dst_img = self.get_roi_image(src_image, rect)
        else:
            points = self.polygon_process.original_coordinate_transformation(points)
            img_crop_width = int(
                max(
                    np.linalg.norm(points[0] - points[1]),
                    np.linalg.norm(points[2] - points[3])))
            img_crop_height = int(
                max(
                    np.linalg.norm(points[0] - points[3]),
                    np.linalg.norm(points[1] - points[2])))
            pts_std = np.float32([[0, 0], [img_crop_width, 0],
                                  [img_crop_width, img_crop_height],
                                  [0, img_crop_height]])
            M = cv2.getPerspectiveTransform(points, pts_std)
            dst_img = cv2.warpPerspective(
                src_image,
                M, (img_crop_width, img_crop_height),
                borderMode=cv2.BORDER_REPLICATE,
                flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img


