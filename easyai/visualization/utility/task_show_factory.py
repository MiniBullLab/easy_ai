#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW
from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger


class TaskShowFactory():

    def __init__(self):
        pass

    def get_task_show(self, task_name):
        task_name = task_name.strip()
        config_args = {'type': task_name}
        result = None
        if REGISTERED_TASK_SHOW.has_class(task_name):
            result = build_from_cfg(config_args, REGISTERED_TASK_SHOW)
        else:
            EasyLogger.error("%s show not exits" % task_name)
        return result
