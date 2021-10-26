#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.BinaryPostProcess)
class BinaryPostProcess(BasePostProcess):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold  # binary class threshold

    def __call__(self, prediction):
        output_count = prediction.size(1)
        assert output_count == 1
        class_indices = (prediction >= self.threshold).to(torch.int32)
        return class_indices, prediction
