#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.data_loader.pose2d.pose2d_dataloader import get_pose2d_train_dataloader
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.pose2d.pose2d_test import Pose2dTest
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.Pose2d_Task)
class Pose2dTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Pose2d_Task)
        self.set_model_param(data_channel=self.train_task_config.data_channel,
                             points_count=len(self.train_task_config.points_count))
        self.set_model(gpu_id=gpu_id)
        self.pose2d_test = Pose2dTest(model_name, gpu_id, self.train_task_config)
        self.best_value = 0
        self.avg_loss = -1

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_value = \
                self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)

        self.freeze_process.freeze_block(self.model,
                                         self.train_task_config.freeze_layer_name,
                                         self.train_task_config.freeze_layer_type)

        self.build_optimizer()

    def train(self, train_path, val_path):
        dataloader = get_pose2d_train_dataloader(train_path,
                                                 self.train_task_config)
        self.total_images = len(dataloader)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_images)
        lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_path)

        self.start_train()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            self.optimizer.zero_grad()
            self.train_epoch(epoch, lr_scheduler, dataloader)
            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def train_epoch(self, epoch, lr_scheduler, dataloader):
        for i, (images, targets) in enumerate(dataloader):
            current_iter = epoch * self.total_images + i
            lr = lr_scheduler.get_lr(epoch, current_iter)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            loss_info = self.compute_backward(images, targets, i)
            self.update_logger(i, self.total_images, epoch, loss_info)

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

    def compute_loss(self, output_list, targets, loss_type=0):
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
            print("compute loss error")
        return loss, loss_info

    def update_logger(self, index, total, epoch, loss_info):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss_info['all_loss']
        loss_info.pop('all_loss')

        self.train_logger.loss_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)

        for key, value in loss_info.items():
            self.train_logger.add_scalar(key, value, step)

        print('Epoch: {}[{}/{}]\t Loss: {:.7f}\t Rate: {:.7f} \t Time: {:.5f}\t'.format(epoch,
                                                                                        index,
                                                                                        total,
                                                                                        loss_value,
                                                                                        lr,
                                                                                        self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_loss_log(epoch)
        if self.train_task_config.is_save_epoch_model:
            save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                           "pose2d_model_epoch_%d.pt" % epoch)
        else:
            save_model_path = self.train_task_config.latest_weights_path
        self.torchModelProcess.save_latest_model(epoch, self.best_value,
                                                 self.model, save_model_path)
        self.save_optimizer(epoch)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.pose2d_test.load_weights(save_model_path)
            precision, average_loss = self.pose2d_test.test(val_path)
            self.pose2d_test.save_test_value(epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            print("Val epoch loss: {}".format(average_loss))

            # save best model
            self.best_value = self.torchModelProcess.save_best_model(precision,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            print("no test!")
