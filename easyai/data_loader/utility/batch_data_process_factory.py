#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger
from easyai.data_loader.utility.dataloader_registry import REGISTERED_BATCH_DATA_PROCESS


class BatchDataProcessFactory():

    def __init__(self):
        pass

    def build_process(self, process_args):
        if process_args is None:
            return None
        func_name = process_args['type'].strip()
        result_func = None
        EasyLogger.debug(process_args)
        if REGISTERED_BATCH_DATA_PROCESS.has_class(func_name):
            result_func = build_from_cfg(process_args, REGISTERED_BATCH_DATA_PROCESS)
        else:
            EasyLogger.error("%s batch data process not exits" % func_name)
        return result_func
