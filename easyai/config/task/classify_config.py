#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.base_name.task_name import TaskName
from easyai.config.utility.image_train_config import ImageTrainConfig
from easyai.config.utility.registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Classify_Task)
class ClassifyConfig(ImageTrainConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Classify_Task)
        # data
        self.class_name = None
        self.save_result_name = None
        # test
        # train
        self.log_name = "classify"
        self.train_data_augment = True

        self.config_path = os.path.join(self.config_save_dir, "classify_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('class_name', None) is not None:
            self.class_name = tuple(config_dict['class_name'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['class_name'] = self.class_name

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
        self.image_size = (224, 224)
        self.data_channel = 3
        self.class_name = ('cls1', 'cls2', 'cls3', 'cls4', 'cls5', 'cls6', 'cls7', 'cls8', 'cls9', 'cls10',
                           'cls11', 'cls12', 'cls13', 'cls14', 'cls15', 'cls16', 'cls17', 'cls18', 'cls19', 'cls20',
                           'cls21', 'cls22', 'cls23', 'cls24', 'cls25', 'cls26', 'cls27', 'cls28', 'cls29', 'cls30',
                           'cls31', 'cls32', 'cls33', 'cls34', 'cls35', 'cls36', 'cls37', 'cls38', 'cls39', 'cls40',
                           'cls41', 'cls42', 'cls43', 'cls44', 'cls45', 'cls46', 'cls47', 'cls48', 'cls49', 'cls50',
                           'cls51', 'cls52', 'cls53', 'cls54', 'cls55', 'cls56', 'cls57', 'cls58', 'cls59', 'cls60',
                           'cls61', 'cls62', 'cls63', 'cls64', 'cls65', 'cls66', 'cls67', 'cls68', 'cls69', 'cls70',
                           'cls71', 'cls72', 'cls73', 'cls74', 'cls75', 'cls76', 'cls77', 'cls78', 'cls79', 'cls80',
                           'cls81', 'cls82', 'cls83', 'cls84', 'cls85', 'cls86', 'cls87', 'cls88', 'cls89', 'cls90',
                           'cls91', 'cls92', 'cls93', 'cls94', 'cls95', 'cls96', 'cls97', 'cls98', 'cls99', 'cls100',
                           )
        self.data_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.data_std = (0.2666410733740041, 0.2666410733740041, 0.2666410733740041)
        self.resize_type = 0
        self.normalize_type = 1
        self.save_result_name = "classify_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.test_batch_size = 1
        self.evaluation_result_name = 'cls_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data_augment = True
        self.train_batch_size = 16
        self.enable_mixed_precision = False
        self.is_save_epoch_model = False
        self.latest_weights_name = 'cls_latest.pt'
        self.best_weights_name = 'cls_best.pt'

        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 200

        self.base_lr = 0.01
        self.optimizer_config = {0: {'optimizer': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[60, 1], [120, 0.2], [160, 0.04], [200, 0.008]],
                                    'warmup_type': 2,
                                    'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.freeze_layer_type = 0
        self.freeze_layer_name = "route_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
