#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.common.task_dataset_process import TaskDataSetProcess


class Box2dDataSetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def resize_polygon(self):
        pass
