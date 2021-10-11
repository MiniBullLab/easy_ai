#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import Levenshtein
from easyai.utility.logger import EasyLogger
from easyai.evaluation.utility.base_evaluation import BaseEvaluation
from easyai.helper.average_meter import AverageMeter
from easyai.helper.data_structure import OCRObject
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION


@REGISTERED_EVALUATION.register_module(EvaluationName.RecognizeTextMetric)
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
            pred_text = pred_text.replace(" ", "")
            if isinstance(target, OCRObject):
                target = target.get_text()
            target = target.replace(" ", "")
            max_length = max(len(pred_text), len(target), 1)
            norm_edit_dis = Levenshtein.distance(pred_text, target) / max_length
            self.edit_distance.update(norm_edit_dis)
            if pred_text == target:
                self.accuracy.update(1)
            else:
                self.accuracy.update(0)

    def get_score(self):
        score = {'accuracy': self.accuracy.avg,
                 'edit_distance': 1 - self.edit_distance.avg}
        self.print_evaluation(score)
        return score

    def print_evaluation(self, score):
        for k, v in score.items():
            EasyLogger.info("{}:{}".format(k, v))
