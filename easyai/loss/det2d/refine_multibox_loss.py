#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.base_multi_loss import BaseMultiLoss
from easyai.loss.utility.box2d_process import torch_corners_box2d_ious, torch_box2d_rect_corner
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


class RefineMultiBoxLoss(BaseMultiLoss):

    def __init__(self, class_number, iou_threshold, input_size,
                 anchor_counts, aspect_ratios,
                 min_sizes, max_sizes=None,
                 is_gaussian=False):
        super().__init__(LossName.RefineMultiBoxLoss, class_number, input_size,
                         anchor_counts, aspect_ratios, min_sizes, max_sizes, is_gaussian)
        self.feature_count = len(anchor_counts)
        self.iou_threshold = iou_threshold
        self.variances = (0.1, 0.2)

    def forward(self, output_list, targets=None):
        output_count = len(output_list) // 2
        anchor_boxes = self.priorbox(self.input_size, feature_sizes)