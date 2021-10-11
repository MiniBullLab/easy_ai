#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.task_result_process import TaskPostProcess


class GenerateImageResultProcess(TaskPostProcess):

    def __init__(self, input_size, post_process_args):
        super().__init__()
        self.post_process_args = post_process_args
        self.input_size = input_size
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction):
        result_image = None
        if prediction is not None:
            result_image = self.process_func(prediction)
        return result_image
