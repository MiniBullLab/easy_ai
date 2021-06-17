#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.gan_train_config import GanTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.OneClass)
class OneClassConfig(GanTrainConfig):

    def __init__(self):
        super().__init__(TaskName.OneClass)

        # data
        self.save_result_name = None

        self.config_path = os.path.join(self.config_save_dir, "one_class_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        self.data = {'image_size': (32, 32),  # W * H
                     'data_channel': 3,
                     'resize_type': 0,
                     'normalize_type': -1,
                     'mean': (0.5, 0.5, 0.5),
                     'std': (0.5, 0.5, 0.5)}

        self.post_process = {'type': 'BinaryPostProcess',
                             'threshold': 0.001}

        self.save_result_name = "one_class_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "OneClassDataset"
        self.val_data['dataset'].update(self.data)

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False

        self.evaluation_result_name = 'one_class_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "OneClassDataset"
        self.train_data['dataset'].update(self.data)

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 64
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['drop_last'] = True

        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'generate_latest.pt'
        self.best_weights_name = 'generate_best.pt'
        self.latest_optimizer_name = "generate_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 0.0002
        self.d_optimizer_config = {0: {'type': 'Adam',
                                       'betas': (0.5, 0.999),
                                       'eps': 1e-08,
                                       'weight_decay': 0}
                                   }

        self.g_optimizer_config = {0: {'type': 'Adam',
                                       'betas': (0.5, 0.999),
                                       'eps': 1e-08,
                                       'weight_decay': 0}
                                   }

        self.d_lr_scheduler_config = {'type': 'MultiStageLR',
                                      'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                      'warmup_type': 0,
                                      'warmup_iters': 1000}

        self.g_lr_scheduler_config = {'type': 'MultiStageLR',
                                      'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                      'warmup_type': 0,
                                      'warmup_iters': 1000}

        self.d_skip_batch_backward = 1
        self.g_skip_batch_backward = 1

        self.accumulated_batches = 1
        self.display = 1

        self.freeze_layer_type = 0
        self.freeze_layer_name = ""
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = ""

