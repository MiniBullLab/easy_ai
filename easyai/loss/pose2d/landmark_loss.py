#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_POSE2D_LOSS


class LandmarkLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.LandmarkLoss)