#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.config.task.polygon2d_config import Polygon2dConfig
from easyai.config.task.rec_text_config import RecognizeTextConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.OCR_Task)
class OCRConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__(TaskName.OCR_Task)

        self.det_config = Polygon2dConfig()
        self.text_config = RecognizeTextConfig()

        self.config_path = os.path.join(self.config_save_dir, "ocr_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        det_dict = config_dict['polygon2d']
        self.det_config.load_data_value(det_dict)
        text_dict = config_dict['rec_text']
        self.text_config.load_data_value(text_dict)

    def save_data_value(self, config_dict):
        det_dict = config_dict.get('polygon2d', None)
        if det_dict is None:
            config_dict['polygon2d'] = {}
        self.det_config.save_data_value(config_dict['polygon2d'])
        text_dict = config_dict.get('rec_text', None)
        if text_dict is None:
            config_dict['rec_text'] = {}
        self.text_config.save_data_value(config_dict['rec_text'])

    def load_test_value(self, config_dict):
        det_dict = config_dict['polygon2d']
        self.det_config.load_test_value(det_dict)
        text_dict = config_dict['rec_text']
        self.text_config.load_test_value(text_dict)

    def save_test_value(self, config_dict):
        det_dict = config_dict.get('polygon2d', None)
        if det_dict is None:
            config_dict['polygon2d'] = {}
        self.det_config.save_test_value(config_dict['polygon2d'])
        text_dict = config_dict.get('rec_text', None)
        if text_dict is None:
            config_dict['rec_text'] = {}
        self.text_config.save_test_value(config_dict['rec_text'])

    def load_train_value(self, config_dict):
        det_dict = config_dict['polygon2d']
        self.det_config.load_train_value(det_dict)
        text_dict = config_dict['rec_text']
        self.text_config.load_train_value(text_dict)

    def save_train_value(self, config_dict):
        det_dict = config_dict.get('polygon2d', None)
        if det_dict is None:
            config_dict['polygon2d'] = {}
        self.det_config.save_train_value(config_dict['polygon2d'])
        text_dict = config_dict.get('rec_text', None)
        if text_dict is None:
            config_dict['rec_text'] = {}
        self.text_config.save_train_value(config_dict['rec_text'])

    def get_data_default_value(self):
        self.det_config.detect2d_class = ('others',)

    def get_test_default_value(self):
        pass

    def get_train_default_value(self):
        pass
