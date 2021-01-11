#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch


class ClassifyResultProcess():

    def __init__(self):
        self.threshold = 0.5  # binary class threshold

    def get_classify_result(self, prediction):
        output_count = prediction.size(1)
        if output_count == 1:
            class_indices = (prediction >= self.threshold).astype(int)
            class_confidence = prediction
        else:
            class_indices = torch.argmax(prediction, dim=1)
            class_confidence = prediction[:, class_indices]
        return class_indices, class_confidence
