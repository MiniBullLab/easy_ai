#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.tasks.pose2d.pose2d_result_process import Pose2dResultProcess


class Pose2dAccuracy():

    def __init__(self, points_count, image_size):
        self.points_count = points_count
        self.image_size = image_size
        self.result_process = Pose2dResultProcess(0, points_count, image_size)
        self.avg_acc = 0
        self.count = 0

    def reset(self):
        self.avg_acc = 0
        self.count = 0

    def numpy_eval(self, prediction, targets):
        heatmap_size = (prediction.shape[-1], prediction[-2])
        coords, maxvals = self.result_process.parse_heatmaps(prediction)
        batch_size = coords.shape[0]
        target_coords = np.zeros((batch_size, self.points_count, 2), dtype=np.float)
        for n in range(batch_size):
            for index in range(self.points_count):
                target_coords[n][index] = targets[index][index] / np.array(self.image_size)
                coords[n][index] = coords[index] / np.array(heatmap_size)
        norm = np.ones((batch_size, 2)) / 10

        dists = self.compute_dists(coords, target_coords, norm)

        idx = list(range(self.points_count))
        acc = np.zeros((len(idx)))

        for i in range(len(idx)):
            acc[i] = self.dist_acc(dists[idx[i]])
            if acc[i] >= 0:
                self.avg_acc = self.avg_acc + acc[i]
                self.count += 1

    def get_score(self):
        self.avg_acc = self.avg_acc / self.count if self.count != 0 else 0
        return self.avg_acc

    def compute_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists, thr=0.5):
        ''' Return percentage below threshold while ignoring values with a -1 '''
        dist_cal = np.not_equal(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
        else:
            return -1
