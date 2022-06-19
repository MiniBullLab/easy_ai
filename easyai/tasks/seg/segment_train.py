#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.tasks.seg.segment_test import SegmentionTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TRAIN_TASK.register_module(TaskName.Segment_Task)
class SegmentionTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Segment_Task)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'],
                             class_number=len(self.train_task_config.segment_class))
        self.set_model(gpu_id=gpu_id)
        self.segment_test = SegmentionTest(model_name, gpu_id, self.train_task_config)

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
        for temp_index, batch_data in enumerate(dataloader):
            current_idx = epoch * self.total_batch_data + temp_index
            lr = lr_scheduler.get_lr(epoch, current_idx)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            loss_info = self.compute_backward(batch_data, temp_index)
            self.update_logger(temp_index, self.total_batch_data, epoch, loss_info)

    def compute_backward(self, batch_data, setp_index):
        # Compute loss, compute gradient, update parameters
        input_datas = self.input_datas_processing(batch_data)
        output_list = self.model(input_datas)
        loss, loss_info = self.compute_loss(output_list, batch_data)

        self.loss_backward(loss)

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_batch_data - 1):
            self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_info['all_loss'] = loss.item()
        return loss_info

    def compute_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        loss_info = dict()
        if loss_count == 1 and output_count == 1:
            output = SegmentResultProcess.output_feature_map_resize(output_list[0],
                                                                    batch_data)
            loss = self.model.lossList[0](output, batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count > 1 and loss_count == output_count:
            loss = self.model.lossList[0](output_list[0], batch_data)
            loss_info = self.model.lossList[0].print_loss_info()
            for k in range(1, loss_count):
                output = SegmentResultProcess.output_feature_map_resize(output_list[k],
                                                                        batch_data)
                loss += self.model.lossList[k](output, batch_data)
                temp_info = self.model.lossList[k].print_loss_info()
                for key, value in temp_info.items():
                    loss_info[key] += value
        else:
            EasyLogger.error("compute loss error")
        return loss, loss_info

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            if self.test_first:
                self.segment_test.create_dataloader(val_path)
                self.test_first = False
            if not self.segment_test.start_test():
                self.torchModelProcess.save_best_model(1, save_model_path,
                                                       self.train_task_config.best_weights_path)
                EasyLogger.warn("no test!")
                return
            self.segment_test.load_weights(save_model_path)
            score, average_loss = self.segment_test.test(epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(score,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            self.torchModelProcess.save_best_model(1, save_model_path,
                                                   self.train_task_config.best_weights_path)
            EasyLogger.warn("%s not exists!" % val_path)
            EasyLogger.warn("no test!")


