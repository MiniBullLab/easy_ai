#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.keypoint2d.keypoint2d_test import KeyPoint2dTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TRAIN_TASK.register_module(TaskName.KeyPoint2d_Task)
class KeyPoints2dTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.KeyPoint2d_Task)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'])
        self.set_model(gpu_id=gpu_id)
        self.keypoints_test = KeyPoint2dTest(model_name, gpu_id, self.train_task_config)
        self.avg_loss = -1

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
        for i, batch_data in enumerate(dataloader):
            current_iter = epoch * self.total_batch_image + i
            lr = lr_scheduler.get_lr(epoch, current_iter)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            if sum([len(x) for x in batch_data['label']]) < 1:  # if no targets continue
                continue
            loss_info = self.compute_backward(batch_data, i)
            self.update_logger(i, self.total_images, epoch, loss_info)

    def compute_backward(self, batch_data, setp_index):
        # Compute loss, compute gradient, update parameters
        input_datas = batch_data['image'].to(self.device)
        output_list = self.model(input_datas)
        loss, loss_info = self.compute_loss(output_list, batch_data)

        self.loss_backward(loss)

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_info['all_loss'] = loss.item()
        return loss_info

    def compute_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        loss_info = {}
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

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            if self.test_first:
                self.classify_test.create_dataloader(val_path)
                self.test_first = False
            if not self.classify_test.start_test():
                EasyLogger.info("no test!")
                return
            self.keypoints_test.load_weights(save_model_path)
            accuracy = self.keypoints_test.test(epoch)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(accuracy, save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            EasyLogger.warn("%s not exists!" % val_path)
            EasyLogger.info("no test!")
