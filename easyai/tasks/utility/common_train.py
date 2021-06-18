#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import sys
import os
import torch
from easyai.tasks.utility.base_train import BaseTrain
from easyai.helper.average_meter import AverageMeter
from easyai.tasks.utility.base_task import DelayedKeyboardInterrupt
from easyai.utility.logger import EasyLogger
try:
    from apex import amp
except ImportError:
    EasyLogger.error("import amp fail!")


class CommonTrain(BaseTrain):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)
        self.optimizer = None
        self.lr_scheduler = None
        self.start_epoch = 0
        self.loss_info_average = dict()

    def load_pretrain_model(self, weights_path):
        if isinstance(weights_path, (list, tuple)):
            if len(weights_path) > 0:
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
            EasyLogger.error("model is not create!")

    def build_lr_scheduler(self):
        self.lr_factory.set_epoch_iteration(self.total_batch_image)
        self.lr_scheduler = self.lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

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
            EasyLogger.error("optimizer is not build!")

    def start_train(self):
        self.model.train()
        self.freeze_process.freeze_bn(self.model,
                                      self.train_task_config.freeze_bn_layer_name,
                                      self.train_task_config.freeze_bn_type)
        self.timer.tic()
        EasyLogger.warn("image count is : %d" % self.total_batch_image)
        assert self.total_batch_image > 0

        for key in self.loss_info_average:
            self.loss_info_average[key].reset()

    def update_logger(self, index, total, epoch, loss_info):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss_info['all_loss']
        loss_info.pop('all_loss')

        self.train_logger.loss_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)

        for key, value in loss_info.items():
            if key not in self.loss_info_average.keys():
                self.loss_info_average[key] = AverageMeter()
            self.loss_info_average[key].update(value)
            self.train_logger.add_scalar(key, self.loss_info_average[key].avg, step)

        info_str = 'Epoch: {}[{}/{}]\t Loss: {:.7f}\t Rate: {:.7f} \t Time: {:.5f}\t'.format(epoch,
                                                                                             index,
                                                                                             total,
                                                                                             loss_value,
                                                                                             lr,
                                                                                             self.timer.toc(True))
        EasyLogger.info(info_str)
        print(info_str)

    def save_train_model(self, epoch):
        with DelayedKeyboardInterrupt():
            if self.train_task_config.is_save_epoch_model:
                save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                               "%s_model_%d.pt" % (self.task_name, epoch))
            else:
                save_model_path = self.train_task_config.latest_weights_path
            self.torchModelProcess.save_latest_model(epoch, self.best_score,
                                                     self.model, save_model_path)

            self.save_optimizer(epoch)
        return save_model_path

    @abc.abstractmethod
    def compute_loss(self, output_list, targets):
        pass

