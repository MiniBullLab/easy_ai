#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.anchor_generator import AnchorGenerator
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


@REGISTERED_DET2D_LOSS.register_module(LossName.RPNLoss)
class RPNLoss(BaseLoss):

    def __init__(self, input_size, class_number, anchor_sizes,
                 aspect_ratios, anchor_strides, fg_iou_threshold,
                 bg_iou_threshold, per_image_sample, positive_fraction):
        super().__init__(LossName.RPNLoss)
        assert len(anchor_strides) == len(
            anchor_sizes
        ), "FPNLoss should have len(ANCHOR_STRIDES) == len(ANCHOR_SIZES)"
        self.input_size = input_size
        self.class_number = class_number
        self.anchor_generator = AnchorGenerator(image_size=input_size,
                                                sizes=anchor_sizes,
                                                aspect_ratios=aspect_ratios,
                                                anchor_strides=anchor_strides)