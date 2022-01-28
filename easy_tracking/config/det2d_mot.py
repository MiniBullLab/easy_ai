#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easy_tracking.utility.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.config.task.detect2d_config import Detect2dConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det2d_Mot_Task)
class DetPose2dConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__(TaskName.Det2d_Mot_Task)

        self.det_config = Detect2dConfig()

        self.config_path = os.path.join(self.config_save_dir, "det2d_mot.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_data_value(det_dict)

    def save_data_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_data_value(config_dict['det2d'])
        pose_dict = config_dict.get('pose2d', None)
        if pose_dict is None:
            config_dict['pose2d'] = {}
        self.pose_config.save_data_value(config_dict['pose2d'])

    def load_test_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_test_value(det_dict)

    def save_test_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_test_value(config_dict['det2d'])

    def load_train_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_train_value(det_dict)

    def save_train_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_train_value(config_dict['det2d'])

    def get_data_default_value(self):
        self.det_config.detect2d_class = ('person',)

    def get_test_default_value(self):
        pass

    def get_train_default_value(self):
        pass
