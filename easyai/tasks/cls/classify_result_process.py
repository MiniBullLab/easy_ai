#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.task_result_process import TaskPostProcess
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.registry import build_from_cfg


class ClassifyResultProcess(TaskPostProcess):

    def __init__(self, post_process_args):
        super().__init__()
        self.post_process_args = post_process_args
        self.process_func = self.build_post_process(post_process_args)

    def post_process(self, prediction):
        class_indices, class_confidence = self.process_func(prediction)
        return class_indices, class_confidence

    def build_post_process(self, post_process_args):
        func_name = post_process_args.strip()
        result_func = None
        if REGISTERED_POST_PROCESS.has_class(func_name):
            result_func = build_from_cfg(post_process_args, REGISTERED_POST_PROCESS)
        else:
            print("%s post process not exits" % func_name)
        return result_func
