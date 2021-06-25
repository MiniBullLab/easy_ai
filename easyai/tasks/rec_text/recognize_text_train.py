#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import traceback
from easyai.utility.logger import EasyLogger
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.rec_text.recognize_text_test import RecognizeTextTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.RecognizeText)
class RecognizeTextTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.RecognizeText)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'],
                             class_number=self.train_task_config.character_count)
        self.set_model(gpu_id=gpu_id, init_type="normal")
        self.test_task = RecognizeTextTest(model_name, gpu_id, self.train_task_config)

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score = \
                self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)
        self.build_optimizer()

    def train(self, train_path, val_path):
        self.create_dataloader(train_path)
        self.build_lr_scheduler()
        self.load_latest_param(self.train_task_config.latest_weights_path)
        self.start_train()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            self.optimizer.zero_grad()
            self.train_epoch(epoch, self.lr_scheduler, self.dataloader)
            self.train_logger.epoch_train_loss_log(epoch)
            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def train_epoch(self, epoch, lr_scheduler, dataloader):
        try:
            for i, (images, targets) in enumerate(dataloader):
                current_iter = epoch * self.total_batch_image + i
                lr = lr_scheduler.get_lr(epoch, current_iter)
                lr_scheduler.adjust_learning_rate(self.optimizer, lr)
                loss_info = self.compute_backward(images, targets, i)
                self.update_logger(i, self.total_images, epoch, loss_info)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss, loss_info = self.compute_loss(output_list, targets)

        self.loss_backward(loss)

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_info['all_loss'] = loss.item()
        return loss_info

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        loss_info = {}
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], targets)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, targets)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count > 1 and loss_count == output_count:
            loss = self.model.lossList[0](output_list[0], targets)
            loss_info = self.model.lossList[0].print_loss_info()
            for k in range(1, loss_count):
                loss += self.model.lossList[k](output_list[k], targets)
                temp_info = self.model.lossList[k].print_loss_info()
                for key, value in temp_info.items():
                    loss_info[key] += value
        else:
            EasyLogger.error("compute loss error")
        return loss, loss_info

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.test_task.load_weights(save_model_path)
            accuracy, average_loss = self.test_task.test(val_path, epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(accuracy,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            EasyLogger.warn("%s not exists!" % val_path)
            EasyLogger.info("no test!")



