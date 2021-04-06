#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.anchor_generator import SSDPriorBoxGenerator
from easyai.loss.det2d.utility.det2d_gt_process import Det2dGroundTruthProcess
from easyai.loss.utility.box2d_process import torch_corners_box2d_ious, torch_box2d_rect_corner
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


class RefineMultiBoxLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.RefineMultiBoxLoss)