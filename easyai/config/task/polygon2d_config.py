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
        self.save_result_name = None

        self.save_result_dir = os.path.join(self.root_save_dir, 'polygon2d_results')

        self.config_path = os.path.join(self.config_save_dir, "polygon2d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('detect2d_class', None) is not None:
            self.detect2d_class = tuple(config_dict['detect2d_class'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['detect2d_class'] = self.detect2d_class

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        self.data = {'image_size': (640, 640),  # W * H
                     'data_channel': 3,
                     'resize_type': 4,
                     'normalize_type': -1,
                     'mean': (0.485, 0.456, 0.406),
                     'std': (0.229, 0.224, 0.225)}

        self.detect2d_class = ("others", )
        self.post_process = {'type': 'DBPostProcess',
                             'threshold': 0.6,
                             'db_threshold': 0.3,
                             'unclip_ratio': 1.5}

        self.model_type = 0
        self.model_config = {'type': 'dbnet',
                             'data_channel': self.data['data_channel'],
                             'class_number': len(self.detect2d_class)}

        self.save_result_name = "polygon2d_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "DetOCRDataSet"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['language'] = ("english",)
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 0
        self.val_data['dataloader']['drop_last'] = False
        self.val_data['dataloader']['collate_fn'] = {"type": "DetOCRDataSetCollate",
                                                     "target_type": 1}

        self.evaluation = {"type": "OCRDetectionMetric",
                           "iou_constraint": 0.5}
        self.evaluation_result_name = 'polygon2d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}

        self.train_data['dataset']['type'] = "DetOCRDataSet"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['language'] = ("english",)
        self.train_data['dataset']['is_augment'] = True

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 16
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 0
        self.train_data['dataloader']['drop_last'] = False
        self.train_data['dataloader']['collate_fn'] = {"type": "DetOCRDataSetCollate",
                                                       "target_type": 0}

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

        self.sparse_config = {'enable_sparse': False,
                              'sparse_lr': 0.00001}

        self.base_lr = 0.001
        self.optimizer_config = {0: {'type': 'Adam',
                                     'weight_decay': 1e-4}
                                 }

        self.lr_scheduler_config = {'type': 'CosineLR',
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
