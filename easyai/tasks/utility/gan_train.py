#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.helper.average_meter import AverageMeter
from easyai.tasks.utility.base_train import BaseTrain


class GanTrain(BaseTrain):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)
        self.d_loss_average = AverageMeter()
        self.g_loss_average = AverageMeter()

        self.d_optimizer_list = []
        self.g_optimizer_list = []
        self.total_batch_image = 0
        self.start_epoch = 0

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
            print("model is not create!")

    def start_train(self):
        self.d_loss_average.reset()
        self.g_loss_average.reset()
        self.model.train()
        self.timer.tic()
