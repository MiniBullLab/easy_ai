#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.SuperResolution_Task)
class SuperResolutionConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.SuperResolution_Task)
        # data
        self.upscale_factor = 1

        self.config_path = os.path.join(self.config_save_dir, "super_resolution_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('upscale_factor', None) is not None:
            self.upscale_factor = int(config_dict['upscale_factor'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['upscale_factor'] = self.upscale_factor

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        self.data = {'image_size': (720, 1024),  # W * H
                     'data_channel': 1,
                     'resize_type': 0,
                     'normalize_type': 0,
                     'mean': (0, ),
                     'std': (1, )}

        self.upscale_factor = 2

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "SuperResolutionDataset"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['upscale_factor'] = self.upscale_factor

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False
        self.val_data['dataloader']['collate_fn'] = {"type": "SuperResolutionDataSetCollate"}

        self.evaluation_result_name = 'sr_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir,
                                                   self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "SuperResolutionDataset"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['upscale_factor'] = self.upscale_factor

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 2
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['drop_last'] = True
        self.train_data['dataloader']['collate_fn'] = {"type": "SuperResolutionDataSetCollate"}

        self.is_save_epoch_model = False
        self.latest_weights_name = 'sr_latest.pt'
        self.best_weights_name = 'sr_best.pt'
        self.latest_optimizer_name = "sr_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 1e-3
        self.optimizer_config = {0: {'type': 'Adam',
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

        self.clip_grad_config = {'enable_clip': False,
                                 'max_norm': 20}

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
