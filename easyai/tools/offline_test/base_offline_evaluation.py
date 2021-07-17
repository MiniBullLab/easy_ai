#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
from easyai.evaluation.utility.evaluation_factory import EvaluationFactory


class BaseOfflineEvaluation():

    def __init__(self):
        self.evaluation_factory = EvaluationFactory()

    @abc.abstractmethod
    def process(self, test_path, target_path):
        pass

    @abc.abstractmethod
    def print_evaluation(self, value):
        pass