#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np


class Pose2dAccuracy():
    """
    Calculate accuracy according to PCK,
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """

    def __init__(self, points_count, image_size):
        self.points_count = points_count
        self.image_size = image_size
        self.points_acc = [0 for _ in range(self.points_count)]
        self.acc_count = [0 for _ in range(self.points_count)]

    def reset(self):
        self.points_acc = [0 for _ in range(self.points_count)]
        self.acc_count = [0 for _ in range(self.points_count)]

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
        norm = np.ones((batch_size, 2)) * np.array(self.image_size) / 10
        dists = self.compute_dists(coords, targets, norm)

        idx = list(range(self.points_count))
        acc = np.zeros((len(idx)))

        for i in range(len(idx)):
            acc[i] = self.dist_acc(dists[idx[i]])
            if acc[i] >= 0:
                self.points_acc[i] = self.points_acc[i] + acc[i]
                self.acc_count[i] += 1

    def get_score(self):
        score = self.print_evaluation()
        return score

    def compute_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 0 and target[n, c, 1] > 0:
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

    def print_evaluation(self):
        print('*************')
        print('Eval Results:')
        all_count = 0
        all_acc = 0
        for index, (acc, count) in enumerate(zip(self.points_acc, self.acc_count)):
            all_count += count
            all_acc += acc
            index_acc = acc / count if count != 0 else 0
            print(("point %d: " % index) + '{:.3f}'.format(index_acc))
        print('**************')
        avg_acc = all_acc / all_count if all_count != 0 else 0
        print('Mean acc = {:.3f}'.format(avg_acc))
        return avg_acc
