#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.average_meter import AverageMeter


class LandmarkAccuracy():

    def __init__(self, points_count):
        self.points_count = points_count
        self.threshold = 0.07
        self.accuracy = AverageMeter()
        self.all_dists = []

    def reset(self):
        self.accuracy.reset()
        self.all_dists = []

    def eval(self, result, targets):
        if not isinstance(result, (list, tuple)):
            result_list = [result]
        else:
            result_list = result
        batch_size = len(result_list)
        coords = np.zeros((batch_size, self.points_count, 2), dtype=np.float)
        for n in range(batch_size):
            points = result_list[n].get_key_points()
            for index in range(self.points_count):
                coords[n][index] = np.array([points[index].x, points[index].y])
        self.numpy_eval(coords, targets)

    def numpy_eval(self, coords, targets):
        batch_size = coords.shape[0]
        norm = np.ones(batch_size.size(0))
        # use bbox to normalize
        for i, gt in enumerate(coords):
            norm[i] = self.get_bboxsize(gt)
        dists = self.compute_dists(coords, targets, norm)
        mean_dists = np.mean(dists, 0)
        acc = np.less_equal(mean_dists, self.threshold).sum() * 1.0 / batch_size
        self.accuracy.update(acc, batch_size)
        self.all_dists.append(dists)

    def get_score(self):
        score = self.print_evaluation()
        return score

    def get_bboxsize(self, coords):
        mins = np.min(coords, 0)
        maxs = np.max(coords, 0)
        return np.sqrt(abs(maxs[0] - mins[0]) * abs(maxs[1] - mins[1]))

    def compute_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                    dists[c, n] = np.linalg.norm(preds[n, c, :] - target[n, c, :]) / normalize[n]
                else:
                    dists[c, n] = 0
        return dists

    def print_evaluation(self):
        print('*************')
        print('Eval Results:')
        print('Mean acc = {:.3f}'.format(self.accuracy.avg))
        dists = np.concatenate(self.all_dists, axis=1)
        errors = np.mean(dists, 0)
        axes1 = np.linspace(0, 1, 1000)
        axes2 = np.zeros(1000)
        for i in range(1000):
            axes2[i] = float((errors < axes1[i]).sum()) / errors.shape[0]
        auc = round(np.sum(axes2[:70]) / .7, 2)
        print('AUC = {:.3f}'.format(auc))
        print('**************')
        return auc
