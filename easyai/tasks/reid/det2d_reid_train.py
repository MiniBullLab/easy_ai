#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.reid.det2d_reid_test import Det2dReidTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TRAIN_TASK.register_module(TaskName.Det2D_REID_TASK)
class Det2dReidTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Det2D_REID_TASK)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'],
                             class_number=len(self.train_task_config.detect2d_class))
        self.set_model(gpu_id=gpu_id)
        self.detect_test = Det2dReidTest(model_name, gpu_id, self.train_task_config)

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
            current_iter = epoch * self.total_batch_data + i
            lr = lr_scheduler.get_lr(epoch, current_iter)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            if sum([len(x) for x in batch_data['label']]) < 1:  # if no targets continue
                EasyLogger.warn("Epoch: {}[{}/{}] has not targets".format(epoch,
                                                                          i,
                                                                          self.total_batch_data))
                continue
            loss_info = self.compute_backward(batch_data, i)
            self.update_logger(i, self.total_batch_data, epoch, loss_info)

    def compute_backward(self, batch_data, setp_index):
        # Compute loss, compute gradient, update parameters
        input_datas = self.input_datas_processing(batch_data)
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

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            if self.test_first:
                self.detect_test.create_dataloader(val_path)
                self.test_first = False
            if not self.detect_test.start_test():
                self.torchModelProcess.save_best_model(1, save_model_path,
                                                       self.train_task_config.best_weights_path)
                EasyLogger.info("no test!")
                return
            self.detect_test.load_weights(save_model_path)
            mAP = self.detect_test.test(epoch)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(mAP, save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            self.torchModelProcess.save_best_model(1, save_model_path,
                                                   self.train_task_config.best_weights_path)
            EasyLogger.warn("%s not exists!" % val_path)
            EasyLogger.info("no test!")