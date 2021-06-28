#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from easyai.utility.logger import EasyLogger


class BaseLoss(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.loss_name = name
        self.loss_info = dict()

    def set_name(self, name):
        self.loss_name = name

    def get_name(self):
        return self.loss_name

    def print_loss_info(self):
        info_str = ''
        for key, value in self.loss_info.items():
            info_str += "%s: %.5f|" % (key, value)
        if info_str:
            EasyLogger.debug(info_str)
        return self.loss_info
