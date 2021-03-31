#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


class OneClassResultProcess():

    def __init__(self, post_prcoess_type):
        self.post_prcoess_type = post_prcoess_type

    def postprocess(self, prediction, threshold=0.0):
        class_indices, class_confidence = self.get_one_class_result(prediction, threshold)
        return class_indices, class_confidence

    def get_one_class_result(self, prediction, conf_thresh):
        class_indices = -1
        class_confidence = 0
        if self.post_prcoess_type == 0:
            class_indices = (prediction >= conf_thresh).astype(int)
            class_confidence = prediction
        return int(class_indices), float(class_confidence)
