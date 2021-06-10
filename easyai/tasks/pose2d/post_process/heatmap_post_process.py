#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import math
import numpy as np
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import DetectionKeyPoint
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.HeatmapPostProcess)
class HeatmapPostProcess(BasePostProcess):

    def __init__(self, input_size, threshold):
        super().__init__()
        self.input_size = input_size
        self.threshold = threshold

    def __call__(self, prediction):
        result = DetectionKeyPoint()
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
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]])
                coords[p] += np.sign(diff) * .25
        valid_point = maxvals > self.threshold
        w_ratio = self.input_size[0] / heatmap_width
        h_ratio = self.input_size[1] / heatmap_height
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
