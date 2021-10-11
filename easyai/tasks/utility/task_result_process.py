#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS
from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger


class TaskPostProcess():

    def __init__(self):
        self.process_func = None
        self.dataset_process = ImageDataSetProcess()

    def set_threshold(self, value):
        assert self.process_func is not None
        self.process_func.set_threshold(value)

    @abc.abstractmethod
    def post_process(self, *args, **kwargs):
        pass

    def build_post_process(self, post_process_args):
        func_name = post_process_args['type'].strip()
        result_func = None
        EasyLogger.debug(post_process_args)
        if REGISTERED_POST_PROCESS.has_class(func_name):
            result_func = build_from_cfg(post_process_args, REGISTERED_POST_PROCESS)
        else:
            EasyLogger.error("%s post process not exits" % func_name)
        return result_func
