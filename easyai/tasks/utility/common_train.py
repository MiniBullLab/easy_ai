#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
import sys
import torch
from easyai.tasks.utility.base_train import BaseTrain
try:
    from apex import amp
except ImportError:
    print("import amp fail!")


class CommonTrain(BaseTrain):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)
        self.optimizer = None
        self.total_batch_image = 0
        self.start_epoch = 0

    def load_pretrain_model(self, weights_path):
        if isinstance(weights_path, (list, tuple)):
            self.torchModelProcess.load_pretain_model(weights_path[0], self.model)
        else:
            self.torchModelProcess.load_pretain_model(weights_path, self.model)

    def build_optimizer(self):
        if self.model is not None:
            self.freeze_process.freeze_block(self.model,
                                             self.train_task_config.freeze_layer_name,
                                             self.train_task_config.freeze_layer_type)
            optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                         self.train_task_config.optimizer_config)
            self.optimizer = self.optimizer_process.get_optimizer(optimizer_args,
                                                                  self.model)
            if self.train_task_config.amp_config['enable_amp']:
                assert 'amp' in sys.modules.keys()
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                            opt_level=
                                                            self.train_task_config.amp_config['opt_level'],
                                                            keep_batchnorm_fp32=
                                                            self.train_task_config.amp_config[
                                                            'keep_batchnorm_fp32'],
                                                            verbosity=0)
                self.torchModelProcess.load_latest_optimizer(self.train_task_config.latest_optimizer_path,
                                                             self.optimizer, amp)
            else:
                self.torchModelProcess.load_latest_optimizer(self.train_task_config.latest_optimizer_path,
                                                             self.optimizer)
        else:
            print("model is not create!")

    def adjust_epoch_optimizer(self, epoch):
        if len(self.train_task_config.optimizer_config) <= 1:
            return
        optimizer_args = self.optimizer_process.get_optimizer_config(epoch,
                                                                     self.train_task_config.optimizer_config)

    def loss_backward(self, loss):
        if self.train_task_config.amp_config['enable_amp']:
            loss = loss / self.train_task_config.accumulated_batches
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def clip_grad(self):
        if self.train_task_config.clip_grad_config['enable_clip']:
            self.print_grad_norm()
            max_norm = float(self.train_task_config.clip_grad_config['max_norm'])
            if self.train_task_config.amp_config['enable_amp']:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def save_optimizer(self, epoch):
        if self.optimizer is not None:
            if self.train_task_config.amp_config['enable_amp']:
                self.torchModelProcess.save_optimizer_state(self.train_task_config.latest_optimizer_path,
                                                            epoch, self.optimizer, amp)
            else:
                self.torchModelProcess.save_optimizer_state(self.train_task_config.latest_optimizer_path,
                                                            epoch, self.optimizer)
        else:
            print("optimizer is not build!")

    def start_train(self):
        self.model.train()
        self.freeze_process.freeze_bn(self.model,
                                      self.train_task_config.freeze_bn_layer_name,
                                      self.train_task_config.freeze_bn_type)
        self.timer.tic()
        assert self.total_batch_image > 0

    @abc.abstractmethod
    def compute_loss(self, output_list, targets):
        pass

