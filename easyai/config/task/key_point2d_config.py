#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.KeyPoint2d_Task)
class KeyPoint2dConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.KeyPoint2d_Task)
        # data
        self.points_class = None
        self.points_count = 0
        self.skeleton = ()
        self.confidence_th = 0
        # test
        # train
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False

        self.config_path = os.path.join(self.config_save_dir, "key_point2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('points_class', None) is not None:
            self.points_class = tuple(config_dict['points_class'])
        if config_dict.get('points_count', None) is not None:
            self.points_count = int(config_dict['points_count'])
        if config_dict.get('skeleton', ()) is not None:
            self.skeleton = tuple(config_dict['skeleton'])
        if config_dict.get('confidence_th', None) is not None:
            self.confidence_th = float(config_dict['confidence_th'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['points_class'] = self.points_class
        config_dict['points_count'] = self.points_count
        config_dict['skeleton'] = self.skeleton
        config_dict['confidence_th'] = self.confidence_th

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])
        if config_dict.get('train_multi_scale', None) is not None:
            self.train_multi_scale = bool(config_dict['train_multi_scale'])
        if config_dict.get('balanced_sample', None) is not None:
            self.balanced_sample = bool(config_dict['balanced_sample'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment
        config_dict['train_multi_scale'] = self.train_multi_scale
        config_dict['balanced_sample'] = self.balanced_sample

    def get_data_default_value(self):
        self.image_size = (640, 352)  # W * H
        self.data_channel = 3
        self.post_prcoess_type = 0

        self.resize_type = 1
        self.normalize_type = 0

        self.points_class = ('bike',)
        self.points_count = 9
        self.confidence_th = 0.5
        self.skeleton = [[1, 2], [2, 4], [4, 3], [3, 1], [1, 5], [5, 6],
                         [6, 8], [8, 7], [7, 5], [7, 3], [8, 4], [6, 2]]

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'key_points2d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.log_name = "keypoint2d"
        self.train_data_augment = True
        self.train_multi_scale = False
        self.balanced_sample = False
        self.train_batch_size = 16
        self.is_save_epoch_model = False
        self.latest_weights_name = 'key_point2d_latest.pt'
        self.best_weights_name = 'key_point2d_best.pt'
        self.latest_optimizer_name = "key_point2d_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 100

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 2e-4
        self.optimizer_config = {0: {'type': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
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

