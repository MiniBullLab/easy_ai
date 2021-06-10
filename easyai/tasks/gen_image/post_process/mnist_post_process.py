#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.MNISTPostProcess)
class MNISTPostProcess(BasePostProcess):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def __call__(self, prediction):
        # 将x的范围由(-1,1)伸缩到(0,1)
        out = 0.5 * (prediction + 1)
        out = out.reshape(self.input_size[0], self.input_size[1])
        out *= 255.0
        out = out.clip(0, 255)
        return np.uint8(out)