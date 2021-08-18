#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from easyai.evaluation.utility.base_evaluation import BaseEvaluation
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION


@REGISTERED_EVALUATION.register_module(EvaluationName.OneClassROC)
class OneClassROC(BaseEvaluation):

    def __init__(self, save_dir):
        super().__init__()
        self.save_index = 0
        self.save_dir = save_dir
        self.scores = np.array([])
        self.gt_labels = np.array([])

    def reset(self):
        self.scores = np.array([])
        self.gt_labels = np.array([])

    def eval(self, result, targets):
        self.scores = np.append(self.scores, result)
        self.gt_labels = np.append(self.gt_labels, targets)

    def get_score(self):
        # Scale score vector between [0, 1]
        min_socre = np.min(self.scores)
        max_score = np.max(self.scores)
        if max_score - min_socre > 0:
            self.scores = (self.scores - min_socre) / (max_score - min_socre)
        # True/False Positive Rates.
        fpr, tpr, _ = roc_curve(self.gt_labels, self.scores)
        roc_auc = auc(fpr, tpr)
        if float(roc_auc) == float("nan") or \
                float(roc_auc) == float("inf") or \
                float(roc_auc) == float("-inf"):
            return 0
        self.draw_roc(fpr, tpr, roc_auc)
        self.print_evaluation(roc_auc)
        return float(roc_auc)

    def draw_roc(self, fpr, tpr, roc_auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        save_name = "ROC_%d.pdf" % self.save_index
        plt.savefig(os.path.join(self.save_dir, save_name))
        plt.close()
        self.save_index = (self.save_index + 1) % 100

    def print_evaluation(self, roc_auc):
        print("AUC(Area Under Curve): {:.5f}".format(roc_auc))
