#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.name_manager.task_name import TaskName
from easyai.config.task.detect2d_config import Detect2dConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det2d_Seg_Task)
class Det2dSegConfig(Detect2dConfig):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Det2d_Seg_Task)
        # data
        self.seg_label_type = None
        self.segment_class = None

        self.seg_post_process = None

        # train
        self.log_name = TaskName.Det2d_Seg_Task
        self.config_path = os.path.join(self.config_save_dir, "det2d_seg_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        super().load_config(config_dict)
        if config_dict.get('seg_label_type', None) is not None:
            self.seg_label_type = int(config_dict['seg_label_type'])
        if config_dict.get('segment_class', None) is not None:
            self.segment_class = list(config_dict['segment_class'])

    def save_data_value(self, config_dict):
        super().save_data_value(config_dict)
        config_dict['seg_label_type'] = self.seg_label_type
        config_dict['segment_class'] = self.segment_class

    def get_data_default_value(self):
        self.data = {'image_size': (640, 352),  # W * H
                     'data_channel': 3,
                     'resize_type': 1,
                     'normalize_type': 0,
                     'mean': (0, 0, 0),
                     'std': (1, 1, 1)}

        self.detect2d_class = ("car", )
        self.segment_class = [('background', '255'),
                              ('lane', '0')]
        self.seg_label_type = 1

        self.post_process = {'type': 'YoloPostProcess',
                             'threshold': 0.5,
                             'nms_threshold': 0.45}

        self.seg_post_process = {'type': 'MaskPostProcess',
                                 'threshold': 0.5}

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "Det2dSegDataset"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['detect2d_class'] = self.detect2d_class
        self.val_data['dataset']['seg_class_name'] = self.segment_class
        self.val_data['dataset']['seg_label_type'] = self.seg_label_type

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False

        self.evaluation_result_name = 'det2d_seg_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)
        self.save_result_dir = os.path.join(self.root_save_dir, 'det2d_seg_results')

    def get_train_default_value(self):
        self.train_data = {'dataloader': {}}
        self.train_data['dataloader']['type'] = "Det2dSegTrainDataloader"
        self.train_data['dataloader']['detect2d_class'] = self.detect2d_class
        self.train_data['dataloader']['seg_class_name'] = self.segment_class
        self.train_data['dataloader']['seg_label_type'] = self.seg_label_type
        self.train_data['dataloader'].update(self.data)
        self.train_data['dataloader']['batch_size'] = 16
        self.train_data['dataloader']['balanced_sample'] = False
        self.train_data['dataloader']['is_augment'] = True

        self.is_save_epoch_model = False
        self.latest_weights_name = 'det2d_seg_latest.pt'
        self.best_weights_name = 'det2d_seg_best.pt'
        self.latest_optimizer_name = "det2d_seg_optimizer.pt"

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
        self.freeze_layer_name = "route_0"
        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"
