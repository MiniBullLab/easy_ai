#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.config.task.detect2d_config import Detect2dConfig
from easyai.config.task.pose2d_config import Pose2dConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det_Pose2d_Task)
class DetPose2dConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__(TaskName.Det_Pose2d_Task)

        self.det_config = Detect2dConfig()
        self.pose_config = Pose2dConfig()

        self.config_path = os.path.join(self.config_save_dir, "det_pose2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_data_value(det_dict)
        pose_dict = config_dict['pose2d']
        self.pose_config.load_data_value(pose_dict)

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
        pose_dict = config_dict['pose2d']
        self.pose_config.load_test_value(pose_dict)

    def save_test_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_test_value(config_dict['det2d'])
        pose_dict = config_dict.get('pose2d', None)
        if pose_dict is None:
            config_dict['pose2d'] = {}
        self.pose_config.save_test_value(config_dict['pose2d'])

    def load_train_value(self, config_dict):
        det_dict = config_dict['det2d']
        self.det_config.load_train_value(det_dict)
        pose_dict = config_dict['pose2d']
        self.pose_config.load_train_value(pose_dict)

    def save_train_value(self, config_dict):
        det_dict = config_dict.get('det2d', None)
        if det_dict is None:
            config_dict['det2d'] = {}
        self.det_config.save_train_value(config_dict['det2d'])
        pose_dict = config_dict.get('pose2d', None)
        if pose_dict is None:
            config_dict['pose2d'] = {}
        self.pose_config.save_train_value(config_dict['pose2d'])

    def get_data_default_value(self):
        self.det_config.detect2d_class = ('person',)

    def get_test_default_value(self):
        pass

    def get_train_default_value(self):
        pass
