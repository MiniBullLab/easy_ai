#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import codecs
import json
from easyai.config.utility.base_config import BaseConfig
from easyai.utility.logger import EasyLogger


class PointCloudTaskConfig(BaseConfig):

    def __init__(self, task_name):
        super().__init__(task_name)
        # data
        self.data = dict()
        self.batch_data_process = None
        self.post_process = None

        self.save_result_path = None

        self.model_config = None

        self.get_base_default_value()

    def load_config(self, config_path):
        if config_path is not None and os.path.exists(config_path):
            self.config_path = config_path

        if os.path.exists(self.config_path):
            with codecs.open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            EasyLogger.info(config_dict)
            self.load_data_value(config_dict)
            self.load_test_value(config_dict)
            self.load_train_value(config_dict)
        else:
            EasyLogger.info("{} not exits".format(self.config_path))

    def save_config(self):
        if not os.path.exists(self.config_save_dir):
            os.makedirs(self.config_save_dir, exist_ok=True)
        config_dict = {}
        self.save_data_value(config_dict)
        self.save_test_value(config_dict)
        self.save_train_value(config_dict)
        with codecs.open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, sort_keys=True, indent=4, ensure_ascii=False)

    def load_pc_data_value(self, config_dict):
        if config_dict.get('data', dict()) is not None:
            self.data = config_dict['data']

        if config_dict.get('post_process', None) is not None:
            self.post_process = config_dict['post_process']

    def save_pc_data_value(self, config_dict):
        if self.data is not None and len(self.data) > 0:
            config_dict['data'] = self.data
        if self.post_process is not None:
            config_dict['post_process'] = self.post_process

    def load_data_value(self, config_dict):
        pass

    def save_data_value(self, config_dict):
        pass

    def load_test_value(self, config_dict):
        pass

    def save_test_value(self, config_dict):
        pass

    def load_train_value(self, config_dict):
        pass

    def save_train_value(self, config_dict):
        pass
