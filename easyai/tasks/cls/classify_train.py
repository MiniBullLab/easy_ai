#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.data_loader.cls.classify_dataloader import get_classify_train_dataloader
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.tasks.utility.base_task import DelayedKeyboardInterrupt
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.cls.classify_test import ClassifyTest
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.Classify_Task)
class ClassifyTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Classify_Task)
        self.set_model_param(data_channel=self.train_task_config.data_channel,
                             class_number=len(self.train_task_config.class_name))
        self.set_model(gpu_id=gpu_id)
        self.classify_test = ClassifyTest(model_name, gpu_id, self.train_task_config)
        self.best_precision = 0

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path is not None and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_precision = \
                self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)

        self.freeze_process.freeze_block(self.model,
                                         self.train_task_config.freeze_layer_name,
                                         self.train_task_config.freeze_layer_type)

        self.build_optimizer()

    def train(self, train_path, val_path):

        dataloader = get_classify_train_dataloader(train_path,
                                                   self.train_task_config)

        self.total_images = len(dataloader)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_images)
        lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_path)

        self.start_train()
        try:
            for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
                self.optimizer.zero_grad()
                self.train_epoch(epoch, lr_scheduler, dataloader)
                save_model_path = self.save_train_model(epoch)
                self.test(val_path, epoch, save_model_path)
        except Exception as e:
            print(e)
            raise e
        finally:
            self.train_logger.close()

    def train_epoch(self, epoch, lr_scheduler, dataloader):
        for index, (images, targets) in enumerate(dataloader):
            current_iter = epoch * self.total_images + index
            lr = lr_scheduler.get_lr(epoch, current_iter)
            lr_scheduler.adjust_learning_rate(self.optimizer, lr)
            loss_value = self.compute_backward(images, targets, index)
            self.update_logger(index, self.total_images, epoch, loss_value)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)

        self.loss_backward(loss)

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) or \
                (setp_index == self.total_images - 1):
            self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def compute_loss(self, output_list, targets, loss_type=0):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], targets)
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, targets)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                loss += self.model.lossList[k](output_list[k], targets)
        else:
            print("compute loss error")
        return loss

    def update_logger(self, index, total, epoch, loss_value):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        self.train_logger.loss_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)

        print('Epoch: {} \t Time: {:.5f}\t'.format(epoch, self.timer.toc(True)))
        print('Epoch: {}[{}/{}]\t Loss: {:.7f}\t Lr: {:.7f} \t'.format(epoch, index, total,
                                                                       loss_value, lr))

    def save_train_model(self, epoch):
        with DelayedKeyboardInterrupt():
            self.train_logger.epoch_train_loss_log(epoch)
            if self.train_task_config.is_save_epoch_model:
                save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                               "cls_model_epoch_%d.pt" % epoch)
            else:
                save_model_path = self.train_task_config.latest_weights_path
            self.torchModelProcess.save_latest_model(epoch, self.best_precision,
                                                     self.model, save_model_path)

            self.save_optimizer(epoch)
        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.classify_test.load_weights(save_model_path)
            precision, average_loss = self.classify_test.test(val_path, epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            # save best model
            self.best_precision = self.torchModelProcess.save_best_model(precision,
                                                                         save_model_path,
                                                                         self.train_task_config.best_weights_path)
        else:
            print("no test!")
