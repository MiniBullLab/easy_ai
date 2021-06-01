#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Landmark)
class LandmarkConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.Landmark)

        # data
        self.pose_class = None
        self.points_count = 0
        self.confidence_th = 0
        self.skeleton = ()
        self.save_result_name = None

        # train
        self.train_data_augment = True
        self.config_path = os.path.join(self.config_save_dir, "landmark_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('points_class', None) is not None:
            self.pose_class = tuple(config_dict['points_class'])
        if config_dict.get('points_count', 0) is not None:
            self.points_count = int(config_dict['points_count'])
        if config_dict.get('confidence_th', 0) is not None:
            self.confidence_th = int(config_dict['confidence_th'])
        if config_dict.get('skeleton', ()) is not None:
            self.skeleton = tuple(config_dict['skeleton'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['pose_class'] = self.pose_class
        config_dict['points_count'] = self.points_count
        config_dict['confidence_th'] = self.confidence_th
        config_dict['skeleton'] = self.skeleton

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None) is not None:
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment

    def get_data_default_value(self):
        self.image_size = (128, 128)
        self.data_channel = 1
        self.data_mean = (104.0, )
        self.data_std = (0.017, )
        self.resize_type = 1
        self.normalize_type = 0
        self.save_result_name = "landmark_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

        self.pose_class = ('face',)
        self.points_count = 68
        self.confidence_th = 0.1
        self.skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                         [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
                         [12, 13], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19],
                         [19, 20], [20, 21], [22, 23], [23, 24],
                         [24, 25], [25, 26], [27, 28], [28, 29], [29, 30], [30, 31],
                         [31, 32], [32, 33], [33, 34], [34, 35], [35, 30],
                         [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
                         [42, 43], [43, 44], [44, 45], [45, 46], [46, 47],
                         [47, 42], [48, 49], [49, 50], [50, 51], [51, 52], [52, 53],
                         [53, 54], [54, 55], [55, 56], [56, 57], [57, 58],
                         [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64],
                         [64, 65], [65, 66], [66, 67], [67, 60]]
        self.post_prcoess_type = 0

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'landmark_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data_augment = True
        self.train_batch_size = 32
        self.is_save_epoch_model = False
        self.latest_weights_name = 'landmark_latest.pt'
        self.best_weights_name = 'landmark_best.pt'

        self.latest_optimizer_name = "landmark_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 100

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.0005
        self.optimizer_config = {0: {'type': 'Adam',
                                     'betas': (0.9, 0.999),
                                     'eps': 1e-08,
                                     'weight_decay': 0.0005}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[10, 0.1], [40, 0.01], [80, 0.001]],
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
