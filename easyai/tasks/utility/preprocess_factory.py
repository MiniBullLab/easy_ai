#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.task_registry import REGISTERED_PREPROCESS
from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger


class PreprocessFactory():

    def __init__(self):
        pass

    def build_preprocess(self, process_args):
        if process_args is None:
            return None
        func_name = process_args['type'].strip()
        result_func = None
        EasyLogger.debug(process_args)
        if REGISTERED_PREPROCESS.has_class(func_name):
            result_func = build_from_cfg(process_args, REGISTERED_PREPROCESS)
        else:
            EasyLogger.error("%s post process not exits" % func_name)
        return result_func
