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
        self.character_count = 0
        self.save_result_name = None
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'rec_text_results')

        self.config_path = os.path.join(self.config_save_dir, "rec_text_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('language', None) is not None:
            self.language = tuple(config_dict['language'])
        if config_dict.get('character_set', None) is not None:
            self.character_set = config_dict['character_set']
        if config_dict.get('character_count', 0) is not None:
            self.character_count = int(config_dict['character_count'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['language'] = self.language
        config_dict['character_set'] = self.character_set
        config_dict['character_count'] = self.character_count

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        current_path = inspect.getfile(inspect.currentframe())
        dir_name = os.path.join(os.path.dirname(current_path), "../character")
        self.character_set = os.path.join(dir_name, "zh_en.txt")
        self.character_count = 6625

        self.data = {'image_size': (320, 32),   # W * H
                     'data_channel': 3,
                     'resize_type': -1,
                     'normalize_type': 1,
                     'mean': (0.5, 0.5, 0.5),
                     'std': (0.5, 0.5, 0.5)}
        self.language = ("english", )
        self.post_process = {'type': 'CTCPostProcess'}

        self.save_result_name = "rec_text_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "RecTextDataSet"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['char_path'] = self.character_set
        self.val_data['dataset']['language'] = ("english", )
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False

        self.evaluation_result_name = 'rec_text_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "RecTextDataSet"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['char_path'] = self.character_set
        self.train_data['dataset']['language'] = ("english",)
        self.train_data['dataset']['is_augment'] = False

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 1
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['drop_last'] = True

        self.log_name = "rec_text"
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
