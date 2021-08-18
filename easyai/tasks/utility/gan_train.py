#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.helper.average_meter import AverageMeter
from easyai.tasks.utility.base_train import BaseTrain
from easyai.utility.logger import EasyLogger


class GanTrain(BaseTrain):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)
        self.d_loss_average = AverageMeter()
        self.g_loss_average = AverageMeter()

        self.d_optimizer_list = []
        self.g_optimizer_list = []
        self.total_batch_image = 0
        self.best_score = 0
        self.start_epoch = 0

        self.test_first = True

    def load_pretrain_model(self, weights_path):
        if isinstance(weights_path, (list, tuple)):
            if len(weights_path) > 0:
                self.torchModelProcess.load_pretain_model(weights_path[0], self.model)
        else:
            self.torchModelProcess.load_pretain_model(weights_path, self.model)

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score \
                = self.torchModelProcess.load_latest_model(latest_weights_path, self.model)
            EasyLogger.debug("Latest value: {} {}".format(self.start_epoch,
                                                          self.best_score))
        self.model = self.torchModelProcess.model_train_init(self.model)
        self.build_optimizer()

    def build_optimizer(self):
        if self.model is not None:
            # self.freeze_process.freeze_block(self.model,
            #                                  self.train_task_config.freeze_layer_name,
            #                                  self.train_task_config.freeze_layer_type)
            d_optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                           self.train_task_config.
                                                                           d_optimizer_config)
            g_optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                           self.train_task_config.
                                                                           g_optimizer_config)
            for d_model in self.model.d_model_list:
                optimizer = self.optimizer_process.get_optimizer(d_optimizer_args,
                                                                 d_model)
                self.d_optimizer_list.append(optimizer)

            for g_model in self.model.g_model_list:
                optimizer = self.optimizer_process.get_optimizer(g_optimizer_args,
                                                                 g_model)
                self.g_optimizer_list.append(optimizer)
        else:
            EasyLogger.error("model is not create!")

    def start_train(self):
        self.d_loss_average.reset()
        self.g_loss_average.reset()
        self.model.train()
        self.timer.tic()
        assert self.total_batch_image > 0

    def save_train_model(self, epoch):
        self.train_logger.add_scalar("train epoch d loss",
                                     self.d_loss_average.avg, epoch)
        self.train_logger.add_scalar("train epoch g loss",
                                     self.g_loss_average.avg, epoch)
        self.d_loss_average.reset()
        self.g_loss_average.reset()

        if self.train_task_config.is_save_epoch_model:
            save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                           "%s_model_%d.pt" % (self.task_name, epoch))
        else:
            save_model_path = self.train_task_config.latest_weights_path
        self.torchModelProcess.save_latest_model(epoch, 0, self.model, save_model_path)

        return save_model_path

    def compute_g_loss(self, output_list, batch_data):
        loss = []
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            result = self.model.g_loss_list[0](output_list[0], batch_data)
            loss.append(result)
        elif loss_count == 1 and output_count > 1:
            result = self.model.g_loss_list[0](output_list, batch_data)
            loss.append(result)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                result = self.model.g_loss_list[k](output_list[k], batch_data)
                loss.append(result)
        else:
            EasyLogger.error("compute generator loss error")
        return loss

    def compute_d_loss(self, output_list, batch_data):
        loss = []
        loss_count = len(self.model.d_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            result = self.model.d_loss_list[0](output_list[0], batch_data)
            loss.append(result)
        elif loss_count == 1 and output_count > 1:
            result = self.model.d_loss_list[0](output_list, batch_data)
            loss.append(result)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                result = self.model.d_loss_list[k](output_list[k], batch_data)
                loss.append(result)
        else:
            EasyLogger.error("compute discriminator loss error")
        return loss
