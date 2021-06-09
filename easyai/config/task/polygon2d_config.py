#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Polygon2d_Task)
class Polygon2dConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.Polygon2d_Task)
        # data
        self.detect2d_class = None
        self.post_process = None
        self.save_result_name = None
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'polygon2d_results')
        # train
        self.train_data_augment = True

        self.config_path = os.path.join(self.config_save_dir, "polygon2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('detect2d_class', None) is not None:
            self.detect2d_class = tuple(config_dict['detect2d_class'])
        if config_dict.get('post_process', None) is not None:
            self.post_process = config_dict['post_process']

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['detect2d_class'] = self.detect2d_class
        config_dict['post_process'] = self.post_process

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment

    def get_data_default_value(self):
        self.image_size = (736, 736)  # W * H
        self.data_channel = 3
        self.detect2d_class = ("others", )
        self.post_process = {'type': 'DBPostProcess',
                             'threshold': 0.3,
                             'unclip_ratio': 1.5}

        self.resize_type = -2
        self.normalize_type = -1
        self.data_mean = (0.485, 0.456, 0.406)
        self.data_std = (0.229, 0.224, 0.225)

        self.save_result_name = "polygon2d_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'polygon2d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.log_name = "detect2d"
        self.train_data_augment = True
        self.train_batch_size = 4
        self.is_save_epoch_model = False
        self.latest_weights_name = 'polygon2d_latest.pt'
        self.best_weights_name = 'polygon2d_best.pt'
        self.latest_optimizer_name = "polygon2d_optimizer.pt"

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
        self.freeze_layer_name = "baseNet_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
