#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import inspect
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.RecognizeText)
class RecognizeTextConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.RecognizeText)
        # data
        self.language = None
        self.character_set = None
        self.post_process = None
        self.save_result_name = None
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'rec_text_results')
        # train
        self.train_data_augment = True

        self.config_path = os.path.join(self.config_save_dir, "rec_text_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('language', None) is not None:
            self.language = config_dict['language']
        if config_dict.get('character_set', None) is not None:
            self.character_set = config_dict['character_set']
        if config_dict.get('post_process', None) is not None:
            self.post_process = config_dict['post_process']

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['language'] = self.language
        config_dict['character_set'] = self.character_set
        config_dict['post_process'] = self.post_process

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)
        if config_dict.get('train_data_augment', None) is not None:
            self.train_data_augment = bool(config_dict['train_data_augment'])

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)
        config_dict['train_data_augment'] = self.train_data_augment

    def get_data_default_value(self):
        self.image_size = (640, 640)  # W * H
        self.data_channel = 3
        self.language = "en"
        current_path = inspect.getfile(inspect.currentframe())
        dir_name = os.path.dirname(current_path)
        self.character_set = os.path.join(dir_name, "/..", "character/en.txt")
        self.post_process = {'type': 'DBPostProcess',
                             'threshold': 0.3,
                             'unclip_ratio': 1.5}

        self.resize_type = -1
        self.normalize_type = 0

        self.save_result_name = "rec_text_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'rec_text_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.log_name = "rec_text"
        self.train_data_augment = True
        self.train_batch_size = 4
        self.is_save_epoch_model = False
        self.latest_weights_name = 'rec_text_latest.pt'
        self.best_weights_name = 'rec_text_best.pt'
        self.latest_optimizer_name = "rec_text_optimizer.pt"

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
