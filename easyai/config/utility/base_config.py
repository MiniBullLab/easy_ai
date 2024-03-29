#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import abc
from easyai.utility.logger import EasyLogger


class BaseConfig():

    def __init__(self, task_name):
        self.task_name = task_name
        self.root_save_dir = EasyLogger.ROOT_DIR
        self.model_save_dir_name = None
        self.snapshot_dir = None
        self.config_save_dir_name = None
        self.config_save_dir = None
        self.config_path = None
        self.log_name = task_name
        self.log_save_path = None

    @abc.abstractmethod
    def load_config(self, config_path):
        pass

    @abc.abstractmethod
    def save_config(self):
        pass

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name

    def get_base_default_value(self):
        self.model_save_dir_name = "snapshot"
        self.config_save_dir_name = "config"

        self.snapshot_dir = os.path.join(self.root_save_dir, self.model_save_dir_name)

        self.config_save_dir = os.path.join(self.root_save_dir, self.config_save_dir_name)

        self.log_save_path = os.path.join(self.root_save_dir, "%s.log" % self.log_name)

        if self.root_save_dir is not None and not os.path.exists(self.root_save_dir):
            os.makedirs(self.root_save_dir, exist_ok=True)
