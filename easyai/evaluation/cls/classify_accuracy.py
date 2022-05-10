#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper.average_meter import AverageMeter
from easyai.evaluation.utility.base_evaluation import BaseEvaluation
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION
from easyai.utility.logger import EasyLogger


@REGISTERED_EVALUATION.register_module(EvaluationName.ClassifyAccuracy)
class ClassifyAccuracy(BaseEvaluation):

    def __init__(self, top_k=(1, 5)):
        super().__init__()
        self.top1 = AverageMeter()
        self.topK = AverageMeter()
        self.param_top = top_k
        self.threshold = 0.5  # binary class threshold
        self.reset()

    def get_k(self):
        return self.param_top

    def torch_eval(self, output, target):
        precision = self.accuracy(output, target, self.param_top)
        batch_size = target.size(0)
        if len(precision) > 1:
            self.top1.update(precision[0].item(), batch_size)
            self.topK.update(precision[1].item(), batch_size)
        else:
            self.top1.update(precision[0].item(), batch_size)

    def numpy_eval(self, outputs, targets):
        maxk = max(self.param_top)
        preds = np.argsort(-outputs)[:maxk]
        targets = targets.repeat(preds.shape, axis=1)
        correct = preds == targets
        res = []
        for k in self.param_top:
            correct_k = correct[:k].sum(0)
            res.append(correct_k.mul_(100))
        if len(res) > 1:
            self.top1.update(res[0], 1)
            self.topK.update(res[1], 1)
        else:
            self.top1.update(res[0], 1)

    def result_eval(self, x, y):
        if x == y:
            self.top1.update(1, 1)
        else:
            self.top1.update(0, 1)

    def reset(self):
        self.top1.reset()
        self.topK.reset()

    def get_top1(self):
        self.print_evaluation()
        return self.top1.avg

    def get_topK(self):
        return self.topK.avg

    def accuracy(self, output, target, top=(1, 5)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(top)
        batch_size = target.size(0)
        class_number = output.size(1)
        if class_number > 1:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = (output >= self.threshold).astype(int)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def print_evaluation(self):
        if max(self.param_top) > 1:
            EasyLogger.info('prec{}: {:.3f} \t prec{}: {:.3f}\t'.format(self.param_top[0],
                                                                        self.top1.avg,
                                                                        self.param_top[1],
                                                                        self.topK.avg))
        else:
            EasyLogger.info('prec1: {:.3f} \t'.format(self.top1.avg))
