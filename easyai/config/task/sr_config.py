#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig


class SuperResolutionConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.SuperResolution_Task)

        # data
        self.upscale_factor = 1
        # test
        self.save_sr_path = os.path.join(self.root_save_dir, 'sr_evaluation.txt')
        # train
        self.log_name = TaskName.SuperResolution_Task
        self.train_data_augment = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'sr_latest.pt'
        self.best_weights_name = 'sr_best.pt'
        self.latest_weights_file = None
        self.best_weights_file = None
        self.accumulated_batches = 1
        self.display = 1

        self.freeze_layer_type = 0
        self.freeze_layer_name = None

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = None

        self.config_path = os.path.join(self.config_save_dir, "super_resolution_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        if config_dict.get('image_size', None) is not None:
            self.image_size = tuple(config_dict['image_size'])
        if config_dict.get('image_channel', None) is not None:
            self.image_channel = int(config_dict['image_channel'])
        if config_dict.get('upscale_factor', None) is not None:
            self.upscale_factor = int(config_dict['upscale_factor'])

    def save_data_value(self, config_dict):
        config_dict['image_size'] = self.image_size
        config_dict['image_channel'] = self.image_channel
        config_dict['upscale_factor'] = self.upscale_factor

    def load_test_value(self, config_dict):
        if config_dict.get('test_batch_size', None) is not None:
            self.test_batch_size = int(config_dict['test_batch_size'])

    def save_test_value(self, config_dict):
        config_dict['test_batch_size'] = self.test_batch_size

    def load_train_value(self, config_dict):
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])
        if config_dict.get('train_batch_size', None) is not None:
            self.train_batch_size = int(config_dict['train_batch_size'])
        if config_dict.get('is_save_epoch_model', None) is not None:
            self.is_save_epoch_model = bool(config_dict['is_save_epoch_model'])
        if config_dict.get('latest_weights_name', None) is not None:
            self.latest_weights_name = str(config_dict['latest_weights_name'])
        if config_dict.get('best_weights_name', None) is not None:
            self.best_weights_name = str(config_dict['best_weights_name'])
        if config_dict.get('max_epochs', None) is not None:
            self.max_epochs = int(config_dict['max_epochs'])
        if config_dict.get('base_lr', None) is not None:
            self.base_lr = float(config_dict['base_lr'])
        if config_dict.get('optimizer_config', None) is not None:
            optimizer_dict = config_dict['optimizer_config']
            self.optimizer_config = {}
            for epoch, value in optimizer_dict.items():
                self.optimizer_config[int(epoch)] = value
        if config_dict.get('lr_scheduler_config', None) is not None:
            self.lr_scheduler_config = config_dict['lr_scheduler_config']
        if config_dict.get('accumulated_batches', None) is not None:
            self.accumulated_batches = int(config_dict['accumulated_batches'])
        if config_dict.get('display', None) is not None:
            self.display = int(config_dict['display'])
        if config_dict.get('freeze_layer_type', None) is not None:
            self.freeze_layer_type = int(config_dict['freeze_layer_type'])
        if config_dict.get('freeze_layer_name', None) is not None:
            self.freeze_layer_name = config_dict['freeze_layer_name']
        if config_dict.get('freeze_bn_type', None) is not None:
            self.freeze_bn_type = int(config_dict['freeze_bn_type'])
        if config_dict.get('freeze_bn_layer_name', None) is not None:
            self.freeze_bn_layer_name = config_dict['freeze_bn_layer_name']

    def save_train_value(self, config_dict):
        config_dict['train_data_augment'] = self.train_data_augment
        config_dict['train_batch_size'] = self.train_batch_size
        config_dict['is_save_epoch_model'] = self.is_save_epoch_model
        config_dict['latest_weights_name'] = self.latest_weights_name
        config_dict['best_weights_name'] = self.best_weights_name
        config_dict['max_epochs'] = self.max_epochs
        config_dict['base_lr'] = self.base_lr
        config_dict['optimizer_config'] = self.optimizer_config
        config_dict['lr_scheduler_config'] = self.lr_scheduler_config
        config_dict['accumulated_batches'] = self.accumulated_batches
        config_dict['display'] = self.display
        config_dict['freeze_layer_type'] = self.freeze_layer_type
        config_dict['freeze_layer_name'] = self.freeze_layer_name
        config_dict['freeze_bn_type'] = self.freeze_bn_type
        config_dict['freeze_bn_layer_name'] = self.freeze_bn_layer_name

    def get_data_default_value(self):
        self.image_size = (720, 1024)  # w * H
        self.image_channel = 1
        self.upscale_factor = 2

    def get_test_default_value(self):
        self.test_batch_size = 1

    def get_train_default_value(self):
        self.train_data_augment = False
        self.train_batch_size = 2
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'sr_latest.pt'
        self.best_weights_name = 'sr_best.pt'
        self.latest_weights_file = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_file = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 1e-3
        self.optimizer_config = {0: {'optimizer': 'Adam',
                                     'betas': (0.9, 0.999),
                                     'eps': 1e-08,
                                     'weight_decay': 0}
                                 }

        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
                                    'warmup_type': 0,
                                    'warmup_iters': 1000}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
