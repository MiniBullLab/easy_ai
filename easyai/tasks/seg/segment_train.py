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
        self.output_process = SegmentResultProcess(self.train_task_config.data['image_size'],
                                                   self.train_task_config.data['resize_type'],
                                                   self.train_task_config.post_process)

        self.segment_test = SegmentionTest(model_name, gpu_id, self.train_task_config)

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score \
                = self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

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
        for temp_index, (images, segments) in enumerate(dataloader):
            current_idx = epoch * self.total_batch_image + temp_index
            lr = lr_scheduler.get_lr(epoch, current_idx)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            loss_info = self.compute_backward(images, segments, temp_index)
            self.update_logger(temp_index, self.total_batch_image, epoch, loss_info)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss, loss_info = self.compute_loss(output_list, targets.to(self.device))

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
            output, target = self.output_process.output_feature_map_resize(output_list[0], targets)
            loss = self.model.lossList[0](output, target)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, targets)
            loss_info = self.model.lossList[0].print_loss_info()
        elif loss_count > 1 and loss_count == output_count:
            output, target = self.output_process.output_feature_map_resize(output_list[0], targets)
            loss = self.model.lossList[0](output, targets)
            loss_info = self.model.lossList[0].print_loss_info()
            for k in range(1, loss_count):
                output, target = self.output_process.output_feature_map_resize(output_list[k], targets)
                loss += self.model.lossList[k](output, target)
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
                EasyLogger.info("no test!")
                return
            self.segment_test.load_weights(save_model_path)
            score, average_loss = self.segment_test.test(epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(score,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            EasyLogger.warn("%s not exists!" % val_path)
            EasyLogger.info("no test!")


