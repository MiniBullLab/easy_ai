#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.task.detect2d_config import Detect2dConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Pose2d_Task)
class Pose2dConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Pose2d_Task)

        # data
        self.pose_class = None
        self.points_count = 0
        self.save_result_name = None

        # train
        self.log_name = "pose2d"

        self.trian_det = False
        self.trian_pose = False
        self.train_data_augment = True

        self.config_path = os.path.join(self.config_save_dir, "pose2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('points_class', None) is not None:
            self.pose_class = tuple(config_dict['points_class'])
        if config_dict.get('points_count', None) is not None:
            self.points_count = tuple(config_dict['points_count'])
        if config_dict.get('post_prcoess_type', None) is not None:
            self.post_prcoess_type = int(config_dict['post_prcoess_type'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['points_class'] = self.points_class
        config_dict['points_count'] = self.points_count
        config_dict['post_prcoess_type'] = self.post_prcoess_type

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None) is not None:
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

        if config_dict.get('trian_det', None) is not None:
            self.trian_det = bool(config_dict['trian_det'])

        if config_dict.get('trian_pose', None) is not None:
            self.trian_pose = bool(config_dict['trian_pose'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment
        config_dict['trian_det'] = self.trian_det
        config_dict['trian_pose'] = self.trian_pose

    def get_data_default_value(self):
        self.image_size = (224, 224)
        self.data_channel = 3
        self.resize_type = 0
        self.normalize_type = -1
        self.save_result_name = "pose2d_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

        self.pose_class = ('person',)
        self.points_count = 17
        self.post_prcoess_type = 0

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'pose2d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.trian_det = True
        self.trian_pose = True

        self.train_data_augment = True
        self.train_batch_size = 16
        self.is_save_epoch_model = False
        self.latest_weights_name = 'pose2d_latest.pt'
        self.best_weights_name = 'pose2d_best.pt'

        self.latest_optimizer_name = "pose2d_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 200

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.1
        self.optimizer_config = {0: {'type': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[60, 1], [120, 0.2], [160, 0.04], [200, 0.008]],
                                    'warmup_type': 2,
                                    'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.clip_grad_config = {'enable_clip': False,
                                 'max_norm': 20}

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"