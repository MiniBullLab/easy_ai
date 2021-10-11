#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.task_result_process import TaskPostProcess


class ClassifyResultProcess(TaskPostProcess):

    def __init__(self, post_process_args):
        super().__init__()
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction):
        if prediction is None:
            return None
        class_indices, class_confidence = self.process_func(prediction)
        return class_indices, class_confidence
