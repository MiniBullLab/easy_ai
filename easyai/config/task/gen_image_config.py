#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.gan_train_config import GanTrainConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.GenerateImage)
class GenerateImageConfig(GanTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.GenerateImage)

        # data
        # train
        self.log_name = TaskName.GenerateImage

        self.config_path = os.path.join(self.config_save_dir, "generate_image_config.json")

        self.get_data_default_value()
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
        self.image_size = (720, 1024)  # w * H
        self.data_channel = 1
        self.resize_type = 0
        self.normalize_type = 0
        self.data_mean = (0, 0, 0)
        self.data_std = (1, 1, 1)

    def get_train_default_value(self):
        self.train_batch_size = 2
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'generate_latest.pt'
        self.best_weights_name = 'generate_best.pt'
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.base_lr = 0.0003
        self.d_optimizer_config = {0: {'optimizer': 'Adam',
                                       'betas': (0.9, 0.999),
                                       'eps': 1e-08,
                                       'weight_decay': 0}
                                   }

        self.g_optimizer_config = {0: {'optimizer': 'Adam',
                                       'betas': (0.9, 0.999),
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

        self.freeze_layer_type = 0
        self.freeze_layer_name = ""
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = ""
