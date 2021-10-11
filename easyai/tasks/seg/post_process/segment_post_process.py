#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.SegmentPostProcess)
class SegmentPostProcess():

    def __init__(self):
        pass

    def __call__(self, prediction):
        result = None
        if prediction.ndim == 3:
            result = np.argmax(prediction, axis=0)
        elif prediction.ndim == 4:
            result = np.argmax(prediction, axis=1)
        return result
