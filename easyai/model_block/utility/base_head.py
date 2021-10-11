#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import abc


class BaseHead(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.block_name = name

    def set_name(self, name):
        self.block_name = name

    def get_name(self):
        return self.block_name

    @abc.abstractmethod
    def get_output_channel(self):
        pass
