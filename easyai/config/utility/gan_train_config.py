#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.config.utility.image_task_config import ImageTaskConfig


class GanTrainConfig(ImageTaskConfig):

    def __init__(self, task_name):
        super().__init__(task_name)
        self.train_data = None
        # train
        self.log_name = task_name
        self.enable_mixed_precision = False
        self.max_epochs = 0
        self.base_lr = 0.0
        self.amp_config = None
        self.d_optimizer_config = None
        self.g_optimizer_config = None
        self.d_lr_scheduler_config = None
        self.g_lr_scheduler_config = None
        self.d_skip_batch_backward = 1
        self.g_skip_batch_backward = 1

        self.is_save_epoch_model = False
        self.latest_weights_name = None
        self.latest_optimizer_name = None
        self.best_weights_name = None
        self.latest_weights_path = None
        self.latest_optimizer_path = None
        self.best_weights_path = None

        self.accumulated_batches = 1
        self.display = 1

        self.clip_grad_config = None

        self.freeze_layer_type = 0
        self.freeze_layer_name = None
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = None

        # test
        self.val_data = None

        if self.snapshot_dir is not None and not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir, exist_ok=True)

    def load_image_train_value(self, config_dict):
        if config_dict.get('train_data', None) is not None:
            self.train_data = config_dict['train_data']
        if config_dict.get('is_save_epoch_model', None) is not None:
            self.is_save_epoch_model = bool(config_dict['is_save_epoch_model'])
        if config_dict.get('latest_weights_name', None) is not None:
            self.latest_weights_name = str(config_dict['latest_weights_name'])
        if config_dict.get('latest_optimizer_name', None) is not None:
            self.latest_optimizer_name = str(config_dict['latest_optimizer_name'])
        if config_dict.get('best_weights_name', None) is not None:
            self.best_weights_name = str(config_dict['best_weights_name'])
        if config_dict.get('max_epochs', None) is not None:
            self.max_epochs = int(config_dict['max_epochs'])
        if config_dict.get('base_lr', None) is not None:
            self.base_lr = float(config_dict['base_lr'])
        if config_dict.get('amp_config', None) is not None:
            self.amp_config = config_dict['amp_config']
        if config_dict.get('d_optimizer_config', None) is not None:
            d_optimizer_config = config_dict['d_optimizer_config']
            self.d_optimizer_config = {}
            for epoch, value in d_optimizer_config.items():
                self.d_optimizer_config[int(epoch)] = value
        if config_dict.get('g_optimizer_config', None) is not None:
            g_optimizer_config = config_dict['g_optimizer_config']
            self.g_optimizer_config = {}
            for epoch, value in g_optimizer_config.items():
                self.g_optimizer_config[int(epoch)] = value
        if config_dict.get('d_lr_scheduler_config', None) is not None:
            self.d_lr_scheduler_config = config_dict['d_lr_scheduler_config']
        if config_dict.get('g_lr_scheduler_config', None) is not None:
            self.g_lr_scheduler_config = config_dict['g_lr_scheduler_config']

        if config_dict.get('d_skip_batch_backward', 1) is not None:
            self.d_skip_batch_backward = int(config_dict['d_skip_batch_backward'])
        if config_dict.get('g_skip_batch_backward', 1) is not None:
            self.g_skip_batch_backward = int(config_dict['g_skip_batch_backward'])

        if config_dict.get('accumulated_batches', 1) is not None:
            self.accumulated_batches = int(config_dict['accumulated_batches'])
        if config_dict.get('display', 1) is not None:
            self.display = int(config_dict['display'])

        if config_dict.get('clip_grad_config', None) is not None:
            self.clip_grad_config = config_dict['clip_grad_config']

        if config_dict.get('freeze_layer_type', None) is not None:
            self.freeze_layer_type = int(config_dict['freeze_layer_type'])
        if config_dict.get('freeze_layer_name', None) is not None:
            self.freeze_layer_name = config_dict['freeze_layer_name']
        if config_dict.get('freeze_bn_type', None) is not None:
            self.freeze_bn_type = int(config_dict['freeze_bn_type'])
        if config_dict.get('freeze_bn_layer_name', None) is not None:
            self.freeze_bn_layer_name = config_dict['freeze_bn_layer_name']

    def save_image_train_value(self, config_dict):
        if self.train_data is not None:
            config_dict['train_data'] = self.train_data
        config_dict['is_save_epoch_model'] = self.is_save_epoch_model
        config_dict['latest_weights_name'] = self.latest_weights_name
        config_dict['latest_optimizer_name'] = self.latest_optimizer_name
        config_dict['best_weights_name'] = self.best_weights_name
        config_dict['max_epochs'] = self.max_epochs
        config_dict['base_lr'] = self.base_lr
        if self.amp_config is not None:
            config_dict['amp_config'] = self.amp_config
        if self.d_optimizer_config is not None:
            config_dict['d_optimizer_config'] = self.d_optimizer_config
        if self.g_optimizer_config is not None:
            config_dict['g_optimizer_config'] = self.g_optimizer_config
        if self.d_lr_scheduler_config is not None:
            config_dict['d_lr_scheduler_config'] = self.d_lr_scheduler_config
        if self.g_lr_scheduler_config is not None:
            config_dict['g_lr_scheduler_config'] = self.g_lr_scheduler_config

        config_dict['d_skip_batch_backward'] = self.d_skip_batch_backward
        config_dict['g_skip_batch_backward'] = self.g_skip_batch_backward

        config_dict['accumulated_batches'] = self.accumulated_batches
        config_dict['display'] = self.display

        if self.clip_grad_config is not None:
            config_dict['clip_grad_config'] = self.clip_grad_config

        config_dict['freeze_layer_type'] = self.freeze_layer_type
        config_dict['freeze_layer_name'] = self.freeze_layer_name
        config_dict['freeze_bn_type'] = self.freeze_bn_type
        config_dict['freeze_bn_layer_name'] = self.freeze_bn_layer_name

    def load_test_value(self, config_dict):
        if config_dict.get('val_data', None) is not None:
            self.val_data = int(config_dict['val_data'])

    def save_test_value(self, config_dict):
        if self.val_data is not None:
            config_dict['val_data'] = self.val_data
