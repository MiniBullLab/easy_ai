#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from shapely.geometry import Polygon
from easyai.utility.logger import EasyLogger
from easyai.evaluation.utility.base_evaluation import BaseEvaluation
from easyai.helper.average_meter import AverageMeter
from easyai.helper.data_structure import OCRObject
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION


@REGISTERED_EVALUATION.register_module(EvaluationName.OCRDetectionMetric)
class OCRDetectionMetric(BaseEvaluation):

    def __init__(self, iou_constraint=0.5):
        super().__init__()
        self.iou_constraint = iou_constraint
        self.all_gt_count = 0
        self.all_det_count = 0
        self.all_matched_count = 0

    def reset(self):
        self.all_gt_count = 0
        self.all_det_count = 0
        self.all_matched_count = 0

    def eval(self, result, targets):
        det_matched = 0
        pairs = []
        det_matched_nums = []
        evaluation_log = ""
        gt_count = len(targets)
        det_count = len(result)

        if gt_count > 0 and det_count > 0:
            output_shape = [gt_count, det_count]
            iou_mat = np.empty(output_shape)
            gt_mat = np.zeros(gt_count, np.int8)
            det_mat = np.zeros(det_count, np.int8)
            for gt_num in range(gt_count):
                for det_num in range(det_count):
                    gt_points = targets[gt_num]
                    ocr_object = result[det_num]
                    det_points = np.array([[p.x, p.y] for p in ocr_object.get_polygon()],
                                          dtype=np.float32)
                    iou_mat[gt_num, det_num] = self.get_intersection_over_union(det_points,
                                                                                gt_points)
            for gt_num in range(gt_count):
                for det_num in range(det_count):
                    if gt_mat[gt_num] == 0 and det_mat[det_num] == 0:
                        if iou_mat[gt_num, det_num] > self.iou_constraint:
                            gt_mat[gt_num] = 1
                            det_mat[det_num] = 1
                            det_matched += 1
                            pairs.append({'gt': gt_num, 'det': det_num})
                            det_matched_nums.append(det_num)
                            evaluation_log += "Match GT #" + str(gt_num) + \
                                              " with Det #" + str(det_num) + "\n"

        self.all_matched_count += det_matched
        self.all_gt_count += gt_count
        self.all_det_count += det_count

    def get_score(self):
        if self.all_gt_count == 0:
            recall = 0
        else:
            recall = float(self.all_matched_count) / self.all_gt_count
        if self.all_det_count == 0:
            precision = 0
        else:
            precision = self.all_matched_count / self.all_det_count

        if (recall + precision) == 0:
            hmean = 0
        else:
            hmean = 2 * (recall * precision) / (recall + precision)
        score = {'precision': precision,
                 'recall': recall,
                 'hmean': hmean}
        self.print_evaluation(score)
        return score

    def print_evaluation(self, score):
        for k, v in score.items():
            EasyLogger.info("{}:{}".format(k, v))

    def get_intersection_over_union(self, det_points, gt_points):
        inter_area = self.get_intersection(det_points, gt_points)
        union_area = self.get_union(det_points, gt_points)
        return inter_area / union_area

    def get_intersection(self, det_points, gt_points):
        # print("metric:", det_points, gt_points)
        return Polygon(det_points).intersection(Polygon(gt_points)).area

    def get_union(self, det_points, gt_points):
        return Polygon(det_points).union(Polygon(gt_points)).area
