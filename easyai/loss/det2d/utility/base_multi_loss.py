#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.anchor_generator import SSDPriorBoxGenerator, PriorBoxGenerator
from easyai.loss.det2d.utility.det2d_gt_process import Det2dGroundTruthProcess


class BaseMultiLoss(BaseLoss):

    def __init__(self, name, class_number,
                 input_size, anchor_counts,
                 aspect_ratios, min_sizes,
                 max_sizes=None, is_gaussian=False):
        super().__init__(name)
        self.class_number = class_number
        self.input_size = input_size
        self.anchor_counts = tuple(anchor_counts)
        self.aspect_ratios = aspect_ratios
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.loc_output = 4
        self.is_gaussian = is_gaussian
        # Just for exp convenience
        if is_gaussian:
            self.loc_output += 4

        self.ssd_priorbox = SSDPriorBoxGenerator(anchor_counts, aspect_ratios,
                                                 min_sizes, max_sizes)
        self.priorbox = PriorBoxGenerator(input_size)
        self.gt_process = Det2dGroundTruthProcess()

    def reshape_box_outputs(self, output_list):
        y_locs = list()
        y_confs = list()
        feature_sizes = list()
        for index, feature_index in enumerate(range(0, len(output_list), 2)):
            loc = output_list[feature_index]
            conf = output_list[feature_index + 1]
            N, C, H, W = loc.size()
            loc = loc.permute(0, 2, 3, 1).contiguous()
            conf = conf.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(N, -1, self.loc_output)
            conf = conf.view(N, -1, self.class_number)
            y_locs.append(loc)
            y_confs.append(conf)
            feature_sizes.append((W, H))
        # loc_preds(tensor): predicted locations, sized [batch_size, 8732, 4]
        # cls_preds(tensor): predicted class confidences, sized [batch_size, 8732, num_classes]
        loc_preds = torch.cat(y_locs, dim=1)
        cls_preds = torch.cat(y_confs, dim=1)
        return loc_preds, cls_preds, feature_sizes
