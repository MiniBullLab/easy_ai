#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import math
import numpy as np
from easyai.helper.dataType import Point2d
from easyai.helper.dataType import DetectionObject
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class Pose2dResultProcess():

    def __init__(self, post_prcoess_type, points_count, image_size):
        self.post_prcoess_type = post_prcoess_type
        self.points_count = points_count
        self.image_size = image_size
        self.dataset_process = ImageDataSetProcess()

    def postprocess(self, prediction, src_size, threshold=0.0):
        if prediction is None:
            return None
        object_pose = self.get_pose_result(prediction, threshold)
        result = self.resize_object_pose(src_size, self.image_size, object_pose)
        return result

    def get_pose_result(self, prediction, conf_thresh):
        result = None
        if self.post_prcoess_type == 0:
            result = self.get_heatmaps_result(prediction, conf_thresh)
        elif self.post_prcoess_type == 1:
            result = self.get_mobile_result(prediction)
        return result

    def get_mobile_result(self, prediction):
        result = DetectionObject()
        x = (prediction.reshape([-1, 2]) + np.array([1.0, 1.0])) / 2.0
        x = x * np.array(self.image_size)
        for value in x:
            point = Point2d(int(value[0]), int(value[1]))
            result.add_key_points(point)
        return result

    def get_heatmaps_result(self, prediction, conf_thresh):
        result = DetectionObject()
        heatmap_height = prediction.shape[1]
        heatmap_width = prediction.shape[2]
        coords, maxvals = self.parse_heatmaps(prediction)
        coords = np.squeeze(coords)
        maxvals = np.squeeze(maxvals)
        heatmaps = np.squeeze(prediction)
        for p in range(coords.shape[0]):
            hm = heatmaps[p]
            px = int(math.floor(coords[p][0] + 0.5))
            py = int(math.floor(coords[p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array([hm[py][px+1] - hm[py][px-1], hm[py+1][px]-hm[py-1][px]])
                coords[p] += np.sign(diff) * .25
        valid_point = maxvals > conf_thresh
        w_ratio = self.image_size[0] / heatmap_width
        h_ratio = self.image_size[1] / heatmap_height
        for index, valid in enumerate(valid_point):
            point = Point2d(-1, -1)
            if valid:
                point.x = int(coords[index][0] * w_ratio)
                point.y = int(coords[index][1] * h_ratio)
            result.add_key_points(point)
        return result

    def parse_heatmaps(self, heatmaps):
        batch_size = 1
        num_points = 0
        width = 0
        if heatmaps.ndim == 4:
            batch_size = heatmaps.shape[0]
            num_points = heatmaps.shape[1]
            width = heatmaps.shape[3]
        elif heatmaps.ndim == 3:
            num_points = heatmaps.shape[0]
            width = heatmaps.shape[2]
        # print(batch_size, num_points, width)
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_points, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_points, 1))
        idx = idx.reshape((batch_size, num_points, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)
        preds *= pred_mask

        return preds, maxvals

    def resize_object_pose(self, src_size, image_size,
                           object_pose):
        result = DetectionObject()
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for value in object_pose.get_key_points():
            if value.x != -1 and value.y != -1:
                x = int((value.x - pad[0] // 2) / ratio)
                y = int((value.y - pad[1] // 2) / ratio)
                point = Point2d(x, y)
            else:
                point = value
            result.add_key_points(point)
        return result
