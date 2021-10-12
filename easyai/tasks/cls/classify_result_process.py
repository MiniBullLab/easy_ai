#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.task_result_process import TaskPostProcess


class ClassifyResultProcess(TaskPostProcess):

    def __init__(self, post_process_args):
        super().__init__()
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction):
        if prediction is None:
            return None
        class_index, class_confidence = self.process_func(prediction)
        # output_count = prediction.size(1)
        # if output_count == 1:
        #     batch_size = prediction.size(0)
        #     class_index = torch.ones(batch_size)
        return class_index[0].cpu().numpy(), \
               class_confidence[0][0].cpu().numpy()
