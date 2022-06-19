#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG

from easy_pc.name_manager.pc_task_name import PCTaskName
from easy_pc.config.utility.pc_train_config import PointCloudTrainConfig


@REGISTERED_TASK_CONFIG.register_module(PCTaskName.PC_Det3d_Task)
class PointCloudDet3dConfig(PointCloudTrainConfig):

    def __init__(self):
        super().__init__(PCTaskName.PC_Det3d_Task)
        # data
        self.detect3d_class = None
        self.save_result_name = None
        # test

        self.config_path = os.path.join(self.config_save_dir, "pc_det3d_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_pc_data_value(config_dict)
        if config_dict.get('detect3d_class', None) is not None:
            self.detect3d_class = tuple(config_dict['detect3d_class'])

    def save_data_value(self, config_dict):
        self.save_pc_data_value(config_dict)
        config_dict['detect3d_class'] = self.detect3d_class

    def load_train_value(self, config_dict):
        self.load_pc_train_value(config_dict)

    def save_train_value(self, config_dict):
        if self.train_data is not None and len(self.train_data) > 0:
            if self.train_data.get('dataloader', None) is not None and \
                    self.train_data['dataloader'].get('detect3d_class', None) is not None:
                self.train_data['dataloader']['detect3d_class'] = self.detect3d_class
            if self.train_data.get('dataset', None) is not None and \
                    self.train_data['dataset'].get('detect3d_class', None) is not None:
                self.train_data['dataset']['detect3d_class'] = self.detect3d_class
        self.save_image_train_value(config_dict)

    def save_test_value(self, config_dict):
        if self.evaluation is not None:
            if self.evaluation.get('detect3d_class', None) is not None:
                self.evaluation['detect3d_class'] = self.detect3d_class
            config_dict['evaluation'] = self.evaluation
        if self.val_data is not None and len(self.val_data) > 0:
            if self.val_data.get('dataloader', None) is not None and \
             self.val_data['dataloader'].get('detect3d_class', None) is not None:
                self.val_data['dataloader']['detect3d_class'] = self.detect3d_class
            if self.val_data.get('dataset', None) is not None and \
               self.val_data['dataset'].get('detect3d_class', None) is not None:
                self.val_data['dataset']['detect3d_class'] = self.detect3d_class
            config_dict['val_data'] = self.val_data

    def get_data_default_value(self):
        self.data = {'image_size': (640, 640),  # W * H
                     'data_channel': 3,
                     'resize_type': 2,
                     'normalize_type': 1,
                     'mean': (0, 0, 0),
                     'std': (1, 1, 1)}
        self.detect3d_class = ('car',)

        self.post_process = {'type': 'YoloPostProcess',
                             'threshold': 0.25,
                             'nms_threshold': 0.45}

        self.save_result_name = "pc_det3d_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "MosaicDet2dDataset"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['detect3d_class'] = self.detect3d_class
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['pin_memory'] = True
        self.val_data['dataloader']['drop_last'] = False
        self.val_data['dataloader']['collate_fn'] = {"type": "MosaicDataSetCollate"}

        self.evaluation = {"type": "DetectionMeanAp",
                           'detect3d_class': self.detect3d_class}
        self.evaluation_result_name = 'pc_det3d_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir,
                                                   self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "MosaicDet2dDataset"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['detect3d_class'] = self.detect3d_class
        self.train_data['dataset']['is_augment'] = True

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 16
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['pin_memory'] = True
        self.train_data['dataloader']['drop_last'] = False
        self.train_data['dataloader']['collate_fn'] = {"type": "MosaicDataSetCollate"}

        self.is_save_epoch_model = False
        self.latest_weights_name = 'pc_det3d_latest.pt'
        self.best_weights_name = 'pc_det3d_best.pt'
        self.latest_optimizer_name = "pc_det3d_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 100

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.01
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

