#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.config.task.classify_config import ClassifyConfig
from easyai.config.task.detect2d_config import Detect2dConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det2D_Classify_REID_TASK)
class Det2dClassifyReidConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__(TaskName.Det2D_Classify_REID_TASK)

        self.det_config = Detect2dConfig()
        self.classify_config = ClassifyConfig()

        self.config_path = os.path.join(self.config_save_dir,
                                        "det2d_classify_reid_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_data_value(det_dict)
        classify_dict = config_dict['classify']
        self.classify_config.load_data_value(classify_dict)

    def save_data_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_data_value(config_dict['det2d'])
        classify_dict = config_dict.get('classify', None)
        if classify_dict is None:
            config_dict['classify'] = {}
        self.classify_config.save_data_value(config_dict['classify'])

    def load_test_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_test_value(det_dict)
        classify_dict = config_dict['classify']
        self.classify_config.load_test_value(classify_dict)

    def save_test_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_test_value(config_dict['det2d'])
        classify_dict = config_dict.get('classify', None)
        if classify_dict is None:
            config_dict['classify'] = {}
        self.classify_config.save_test_value(config_dict['classify'])

    def load_train_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_train_value(det_dict)
        text_dict = config_dict['classify']
        self.classify_config.load_train_value(text_dict)

    def save_train_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_train_value(config_dict['det2d'])
        classify_dict = config_dict.get('classify', None)
        if classify_dict is None:
            config_dict['classify'] = {}
        self.classify_config.save_train_value(config_dict['classify'])

    def get_data_default_value(self):

        self.det_config.data = {'image_size': (640, 640),  # W * H
                                'data_channel': 3,
                                'resize_type': 2,
                                'normalize_type': 1,
                                'mean': (0, 0, 0),
                                'std': (1, 1, 1)}
        self.det_config.detect2d_class = ('car',)

        self.det_config.model_type = 0
        self.det_config.model_config = {'type': 'yolov5s',
                                        'data_channel': self.det_config.data['data_channel'],
                                        'class_number': len(self.det_config.detect2d_class)}

        self.classify_config.data = {'image_size': (64, 128),  # W * H
                                     'data_channel': 3,
                                     'resize_type': 1,
                                     'normalize_type': -1,
                                     'mean': (0.485, 0.456, 0.406),
                                     'std': (0.229, 0.224, 0.225)}

        self.classify_config.model_type = 0
        self.classify_config.model_config = {'type': 'DeepSortNet',
                                             'data_channel': self.classify_config.data['data_channel'],
                                             'class_number': 751,
                                             "reid": 512}

    def get_test_default_value(self):
        pass

    def get_train_default_value(self):
        pass


