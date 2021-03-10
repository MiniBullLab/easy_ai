#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.anchor_generator import AnchorGenerator
from easyai.loss.det2d.utility.matcher import Matcher
from easyai.loss.det2d.utility.select_positive_negative_sampler import SelectPositiveNegativeSampler
from easyai.loss.det2d.utility.box_coder import BoxCoder
from easyai.loss.det2d.utility.rpn_postprocess import RPNPostProcessor
from easyai.tasks.utility.box2d_process import torch_corners_box2d_ious
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


class FastRCNNLoss(BaseLoss):

    def __init__(self, input_size, anchor_sizes,
                 aspect_ratios, anchor_strides,
                 fg_iou_threshold=0.5, bg_iou_threshold=0.5,
                 per_image_sample=256, positive_fraction=0.5):
        super().__init__(LossName.FastRCNNLoss)