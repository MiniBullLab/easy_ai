#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.GanomalyPostProcess)
class GanomalyPostProcess(BasePostProcess):

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def __call__(self, prediction):
        output_count = prediction.size
        assert output_count == 1
        class_indices = (prediction >= self.threshold).astype(int)
        return class_indices[0], prediction[0]
