#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG
from easy_pc.name_manager.pc_task_name import PCTaskName
from easy_pc.config.utility.pc_train_config import PointCloudTrainConfig


@REGISTERED_TASK_CONFIG.register_module(PCTaskName.PC_Classify_Task)
class PointCloudClassifyConfig(PointCloudTrainConfig):

    def __init__(self):
        super().__init__(PCTaskName.PC_Classify_Task)
        # data
        self.class_name = None
        self.save_result_name = None

        self.config_path = os.path.join(self.config_save_dir, "pc_classify_config.json")

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

    def load_train_value(self, config_dict):
        self.load_image_train_value(config_dict)

    def save_train_value(self, config_dict):
        self.save_image_train_value(config_dict)

    def get_data_default_value(self):
        self.data = {'point_features': 3}
        self.class_name = ('cls1', 'cls2', 'cls3', 'cls4', 'cls5', 'cls6', 'cls7', 'cls8', 'cls9', 'cls10',
                           'cls11', 'cls12', 'cls13', 'cls14', 'cls15', 'cls16', 'cls17', 'cls18', 'cls19', 'cls20',
                           'cls21', 'cls22', 'cls23', 'cls24', 'cls25', 'cls26', 'cls27', 'cls28', 'cls29', 'cls30',
                           'cls31', 'cls32', 'cls33', 'cls34', 'cls35', 'cls36', 'cls37', 'cls38', 'cls39', 'cls40')
        self.save_result_name = "pc_classify_result.txt"
        self.save_result_path = os.path.join(self.root_save_dir, self.save_result_name)

        self.post_process = {'type': 'MaxPostProcess'}

    def get_test_default_value(self):
        self.val_data = {'dataset': {},
                         'dataloader': {}}
        self.val_data['dataset']['type'] = "ClassifyPointCloudDataSet"
        self.val_data['dataset'].update(self.data)
        self.val_data['dataset']['is_augment'] = False

        self.val_data['dataloader']['type'] = "DataLoader"
        self.val_data['dataloader']['batch_size'] = 1
        self.val_data['dataloader']['shuffle'] = False
        self.val_data['dataloader']['num_workers'] = 8
        self.val_data['dataloader']['drop_last'] = False
        self.val_data['dataloader']['collate_fn'] = {"type": "ClassifyPointCloudDataSetCollate"}

        self.evaluation = {"type": "ClassifyAccuracy",
                           'top_k': (1,)}
        self.evaluation_result_name = 'cls_evaluation.txt'
        self.evaluation_result_path = os.path.join(self.root_save_dir, self.evaluation_result_name)

    def get_train_default_value(self):
        self.train_data = {'dataset': {},
                           'dataloader': {}}
        self.train_data['dataset']['type'] = "ClassifyPointCloudDataSet"
        self.train_data['dataset'].update(self.data)
        self.train_data['dataset']['is_augment'] = True

        self.train_data['dataloader']['type'] = "DataLoader"
        self.train_data['dataloader']['batch_size'] = 4
        self.train_data['dataloader']['shuffle'] = True
        self.train_data['dataloader']['num_workers'] = 8
        self.train_data['dataloader']['drop_last'] = True
        self.train_data['dataloader']['collate_fn'] = {"type": "ClassifyPointCloudDataSetCollate"}

        self.is_save_epoch_model = False
        self.latest_weights_name = 'pc_cls_latest.pt'
        self.best_weights_name = 'pc_cls_best.pt'

        self.latest_optimizer_name = "pc_cls_optimizer.pt"

        self.latest_optimizer_path = os.path.join(self.snapshot_dir, self.latest_optimizer_name)
        self.latest_weights_path = os.path.join(self.snapshot_dir, self.latest_weights_name)
        self.best_weights_path = os.path.join(self.snapshot_dir, self.best_weights_name)

        self.max_epochs = 200

        self.amp_config = {'enable_amp': False,
                           'opt_level': 'O1',
                           'keep_batchnorm_fp32': True}

        self.base_lr = 0.001
        self.optimizer_config = {0: {'type': 'SGD',
                                     'momentum': 0.9,
                                     'weight_decay': 5e-4}
                                 }
        self.lr_scheduler_config = {'type': 'MultiStageLR',
                                    'lr_stages': [[60, 1], [120, 0.2], [160, 0.04], [200, 0.008]],
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
