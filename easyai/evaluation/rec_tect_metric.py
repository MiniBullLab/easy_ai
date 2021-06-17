#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.evaluation.base_evaluation import BaseEvaluation
from easyai.helper.average_meter import AverageMeter
import Levenshtein


class RecognizeTextMetric(BaseEvaluation):

    def __init__(self):
        super().__init__()
        self.accuracy = AverageMeter()
        self.edit_distance = AverageMeter()

    def reset(self):
        self.accuracy.reset()
        self.edit_distance.reset()

    def eval(self, result, targets):
        for pred, target in zip(result, targets):
            pred_text = pred.get_text()
            max_length = max(len(pred_text), len(target))
            norm_edit_dis = Levenshtein.distance(pred_text, target) / max_length
            self.edit_distance.update(norm_edit_dis)
            if pred_text == target:
                self.accuracy.update(1)
            else:
                self.accuracy.update(0)

    def get_score(self):
        score = {'accuracy': self.accuracy.avg,
                 'edit_distance': self.edit_distance}
        self.print_evaluation(score)
        return score

    def print_evaluation(self, score):
        for k, v in score.items():
            print(k, v)
