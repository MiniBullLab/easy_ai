#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import sys
import os
import torch
from easyai.tasks.utility.base_train import BaseTrain
from easyai.helper.average_meter import AverageMeter
from easyai.name_manager.loss_name import LossName
from easyai.tasks.utility.base_task import DelayedKeyboardInterrupt
from easyai.utility.logger import EasyLogger
try:
    from apex import amp
except ImportError:
    EasyLogger.warn("import amp fail!")


class CommonTrain(BaseTrain):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)
        self.optimizer = None
        self.lr_scheduler = None
        self.best_score = -1
        self.start_epoch = 0
        self.test_first = True
        self.loss_info_average = dict()

    def load_pretrain_model(self, weights_path):
        if isinstance(weights_path, (list, tuple)):
            if len(weights_path) > 0:
                self.torchModelProcess.load_pretain_model(weights_path[0], self.model)
        else:
            self.torchModelProcess.load_pretain_model(weights_path, self.model)

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path is not None and os.path.exists(latest_weights_path):
            try:
                self.start_epoch, self.best_score = \
                    self.torchModelProcess.load_latest_model(latest_weights_path, self.model)
                EasyLogger.debug("Latest value: {} {}".format(self.start_epoch,
                                                              self.best_score))
            except Exception as err:
                # os.remove(weight_path)
                self.torchModelProcess.load_pretain_model(latest_weights_path, self.model)
                EasyLogger.warn(err)

        self.model = self.torchModelProcess.model_train_init(self.model)
        self.build_optimizer()

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
        self.lr_factory.set_epoch_iteration(self.total_batch_data)
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
            loss = loss / self.train_task_config.accumulated_batches
            loss.backward()
        loss_count = len(self.model.lossList)
        if loss_count == 1:
            if self.model.lossList[0].get_name() == LossName.CenterCrossEntropy2dLoss or \
                    self.model.lossList[0].get_name() == LossName.CenterCTCLoss:
                lr = self.optimizer.param_groups[0]['lr']
                for param in self.model.lossList[0].center_loss.parameters():
                    # lr_center is learning rate for center loss, e.g. lr_center = 0.5
                    param.grad.data *= (self.model.lossList[0].lr_center / (self.model.lossList[0].alpha * lr))

        if self.train_task_config.sparse_config.get('enable_sparse', None):
            sparse_lr = self.train_task_config.sparse_config['sparse_lr']
            self.optimize_bn.update_bn(self.model, sparse_lr)

    def clip_grad(self):
        if self.train_task_config.clip_grad_config['enable_clip']:
            # self.print_grad_norm()
            max_norm = float(self.train_task_config.clip_grad_config['max_norm'])
            if self.train_task_config.amp_config['enable_amp']:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def L2_regularization(self, loss, lambda_alpha=0.0002):
        l2_alpha = 0.0
        if self.model is not None:
            for name, param in self.model.named_parameters():
                if "alpha" in name:
                    l2_alpha += torch.pow(param, 2)
            loss += lambda_alpha * l2_alpha
        else:
            EasyLogger.error("model not exists")

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
        EasyLogger.info("Train image count is : %d" % self.total_batch_data)
        assert self.total_batch_data > 0, EasyLogger.error("no train dataset")

        for key in self.loss_info_average:
            self.loss_info_average[key].reset()

    def update_logger(self, index, total, epoch, loss_info):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss_info['all_loss']
        loss_info.pop('all_loss')
        if loss_value != float("inf") and loss_value != float("nan"):
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

    def compute_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        loss_info = dict()
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count > 1 and loss_count == output_count:
            loss = self.model.lossList[0](output_list[0], batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
            for k in range(1, loss_count):
                loss += self.model.lossList[k](output_list[k], batch_data)
                temp_info = self.model.lossList[k].print_loss_info()
                for key, value in temp_info.items():
                    loss_info[key] += value
        else:
            EasyLogger.error("compute loss error")
        return loss, loss_info
