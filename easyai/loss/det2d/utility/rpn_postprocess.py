#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import torch


class RPNPostProcessor(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def reshape_outputs(self, box_cls, box_regression):
        result_box_cls = []
        result_box_regression = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for box_cls_per_level, box_regression_per_level in zip(
                box_cls, box_regression
        ):
            N, AxC, H, W = box_cls_per_level.shape
            Ax4 = box_regression_per_level.shape[1]
            A = Ax4 // 4
            C = AxC // A

            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)

            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)

            result_box_cls.append(box_cls_per_level)
            result_box_regression.append(box_regression_per_level)
        return result_box_cls, result_box_regression

    def forward(self, anchors, objectness, box_regression):
        pass
