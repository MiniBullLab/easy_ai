#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG
from easyai.utility.registry import build_from_cfg


class ConfigFactory():

    def __init__(self):
        pass

    def get_config(self, task_name, config_path=None):
        task_name = task_name.strip()
        result = None
        config_args = {'type': task_name}
        if REGISTERED_TASK_CONFIG.has_class(task_name):
            result = build_from_cfg(config_args, REGISTERED_TASK_CONFIG)
            result.load_config(config_path)
        else:
            print("%s task config not exits" % task_name)
        return result

    def save(self, task_config):
        task_config.save_config()
