#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.MaskPostProcess)
class MaskPostProcess():

    def __init__(self, threshold=0.5):
        self.threshold = threshold  # binary class threshold

    def __call__(self, prediction):
        assert prediction.ndim == 2
        result = (prediction >= self.threshold).astype(int)
        return result
