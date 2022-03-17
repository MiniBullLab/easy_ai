#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det2D_REID_TASK)
class Det2dReidConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.Det2D_REID_TASK)
        # data
        self.detect2d_class = None
        self.save_result_name = None
        # test
        self.save_result_dir = os.path.join(self.root_save_dir, 'det2d_reid_results')

        self.config_path = os.path.join(self.config_save_dir, "det2d_reid_config.json")

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
        if self.train_data is not None and len(self.train_data) > 0:
            if self.train_data.get('dataloader', None) is not None and \
                    self.train_data['dataloader'].get('detect2d_class', None) is not None:
                self.train_data['dataloader']['detect2d_class'] = self.detect2d_class
            if self.train_data.get('dataset', None) is not None and \
                    self.train_data['dataset'].get('detect2d_class', None) is not None:
                self.train_data['dataset']['detect2d_class'] = self.detect2d_class
        self.save_image_train_value(config_dict)

    def save_test_value(self, config_dict):
        if self.evaluation is not None:
            if self.evaluation.get('detect2d_class', None) is not None:
                self.evaluation['detect2d_class'] = self.detect2d_class
            config_dict['evaluation'] = self.evaluation
        if self.val_data is not None and len(self.val_data) > 0:
            if self.val_data.get('dataloader', None) is not None and \
             self.val_data['dataloader'].get('detect2d_class', None) is not None:
                self.val_data['dataloader']['detect2d_class'] = self.detect2d_class
            if self.val_data.get('dataset', None) is not None and \
               self.val_data['dataset'].get('detect2d_class', None) is not None:
                self.val_data['dataset']['detect2d_class'] = self.detect2d_class
            config_dict['val_data'] = self.val_data

    def get_data_default_value(self):
        self.data = {'image_size': (640, 640),  # W * H
                     'data_channel': 3,
                     'resize_type': 2,
                     'normalize_type': 1,
                     'mean': (0, 0, 0),
                     'std': (1, 1, 1)}
        self.detect2d_class = ('car',)

        self.post_process = None

        self.model_type = 0
        self.model_config = {'type': 'FairMOTNet',
                             'data_channel': self.data['data_channel'],
                             'class_number': len(self.detect2d_class),
                             "reid": 64,
                             "max_id": 5984,
                             "backbone_name": "yolov5s_old_bockbone"}

        self.save_result_name = "det2d_reid_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "Det2dReidDataset"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['detect2d_class'] = self.detect2d_class
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['pin_memory'] = True
        self.val_data['dataloader']['drop_last'] = False
        self.val_data['dataloader']['collate_fn'] = {"type": "Det2dReidDatasetCollate"}

        self.evaluation = {"type": "ReidDetectionMeanAp",
                           'detect2d_class': self.detect2d_class}
        self.evaluation_result_name = 'det2d_reid_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "Det2dReidDataset"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['detect2d_class'] = self.detect2d_class
        self.train_data['dataset']['is_augment'] = True

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 8
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['pin_memory'] = True
        self.train_data['dataloader']['drop_last'] = True
        self.train_data['dataloader']['collate_fn'] = {"type": "Det2dReidDatasetCollate"}

        self.is_save_epoch_model = False
        self.latest_weights_name = 'det2d_reid_latest.pt'
        self.best_weights_name = 'det2d_reid_best.pt'
        self.latest_optimizer_name = "det2d_reid_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 50

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.001
        # self.optimizer_config = {0: {'type': 'Adam',
        #                              'weight_decay': 5e-4}
        #                          }
        self.optimizer_config = {0: {'type': 'SGD',
                                     'momentum': 0.937,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'CosineLR',
                                    'warmup_type': 2,
                                    'warmup_iters': 3}
        # self.lr_scheduler_config = {'type': 'MultiStageLR',
        #                             'lr_stages': [[50, 1], [70, 0.1], [100, 0.01]],
        #                             'warmup_type': 2,
        #                             'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.clip_grad_config = {'enable_clip': False,
                                 'max_norm': 20}

        self.freeze_layer_type = 0
        self.freeze_layer_name = "baseNet_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"

