#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
import os
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.common_train_config import CommonTrainConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Segment_Task)
class SegmentionConfig(CommonTrainConfig):

    def __init__(self):
        super().__init__(TaskName.Segment_Task)
        # data
        self.seg_label_type = None
        self.segment_class = None
        self.save_result_dir_name = "segment_results"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_dir_name)

        self.config_path = os.path.join(self.config_save_dir, "segmention_config.json")

        self.get_data_default_value()
        self.get_test_default_value()
        self.get_train_default_value()

    def load_data_value(self, config_dict):
        self.load_image_data_value(config_dict)
        if config_dict.get('seg_label_type', None) is not None:
            self.seg_label_type = int(config_dict['seg_label_type'])
        if config_dict.get('segment_class', None) is not None:
            self.segment_class = list(config_dict['segment_class'])

    def save_data_value(self, config_dict):
        self.save_image_data_value(config_dict)
        config_dict['seg_label_type'] = self.seg_label_type
        config_dict['segment_class'] = self.segment_class

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        self.data = {'image_size': (512, 448),  # W * H
                     'data_channel': 3,
                     'resize_type': 1,
                     'normalize_type': 0,
                     'mean': (0, 0, 0),
                     'std': (1, 1, 1)}

        self.seg_label_type = 1
        self.segment_class = [('background', '255'),
                              ('lane', '0')]

        self.post_process = {'type': 'MaskPostProcess',
                             'threshold': 0.5}

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "SegmentDataset"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['class_names'] = self.segment_class
        self.val_data['dataset']['label_type'] = self.seg_label_type
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False

        self.evaluation_result_name = 'seg_evaluation.txt'
        self.evaluation_result_path = os.path.join(EasyLogger.ROOT_DIR,
                                                   self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "SegmentDataset"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['class_names'] = self.segment_class
        self.train_data['dataset']['label_type'] = self.seg_label_type
        self.train_data['dataset']['is_augment'] = True

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 1
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['drop_last'] = True

        self.is_save_epoch_model = False
        self.latest_weights_name = 'seg_latest.pt'
        self.best_weights_name = 'seg_best.pt'
        self.latest_optimizer_name = "seg_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)
        self.max_epochs = 100

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.001
        self.optimizer_config = {0: {'type': 'RMSprop',
                                     'alpha': 0.9,
                                     'eps': 1e-08,
                                     'weight_decay': 0}
                                 }

        self.lr_scheduler_config = {'type': 'CosineLR',
                                    'warmup_type': 2,
                                    'warmup_iters': 5}
        self.accumulated_batches = 1
        self.display = 20

        self.clip_grad_config = {'enable_clip': False,
                                 'max_norm': 20}

        self.freeze_layer_type = 2
        self.freeze_layer_name = "down_invertedResidual_11"

        self.freeze_bn_type = 0
        self.freeze_bn_layer_name = "route_0"


