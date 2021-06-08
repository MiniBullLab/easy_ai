#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.MaxPostProcess)
class MaxPostProcess():

    def __init__(self):
        pass

    def __call__(self, prediction):
        output_count = prediction.size(1)
        assert output_count > 1
        class_indices = torch.argmax(prediction, dim=1)
        class_confidence = prediction[:, class_indices]
        return class_indices, class_confidence
