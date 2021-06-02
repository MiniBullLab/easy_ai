#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.cls.ce2d_loss import CrossEntropy2dLoss
from easyai.loss.cls.ce2d_loss import BinaryCrossEntropy2dLoss
from easyai.loss.utility.registry import REGISTERED_SEG_LOSS


class DBLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.DBLoss)