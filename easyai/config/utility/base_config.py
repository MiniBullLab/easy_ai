#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import abc


class BaseConfig():

    ROOT_DIR = "./.easy_log"

    def __init__(self, task_name):
        self.task_name = task_name
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

        self.snapshot_dir = os.path.join(self.ROOT_DIR, self.model_save_dir_name)

        self.config_save_dir = os.path.join(self.ROOT_DIR, self.config_save_dir_name)

        self.log_save_path = os.path.join(self.ROOT_DIR, "%s.log" % self.log_name)

        if self.ROOT_DIR is not None and not os.path.exists(self.ROOT_DIR):
            os.makedirs(self.ROOT_DIR, exist_ok=True)
