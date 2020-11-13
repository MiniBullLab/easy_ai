#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_train_dataloader
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.tasks.utility.base_train import BaseTrain
from easyai.tasks.sr.super_resolution_test import SuperResolutionTest
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.SuperResolution_Task)
class SuperResolutionTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.SuperResolution_Task)

        self.model_args['upscale_factor'] = self.train_task_config.upscale_factor
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        self.device = self.torchModelProcess.get_device()

        self.sr_test = SuperResolutionTest(cfg_path, gpu_id, config_path)

        self.optimizer = None
        self.total_images = 0
        self.optimizer = None
        self.start_epoch = 0
        self.best_score = 0

    def load_latest_param(self, latest_weights_path):
        checkpoint = None
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score \
                = self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)

        self.freeze_process.freeze_block(self.model,
                                         self.train_task_config.freeze_layer_name,
                                         self.train_task_config.freeze_layer_type)

        optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                     self.train_task_config.optimizer_config)
        self.optimizer = self.optimizer_process.get_optimizer(optimizer_args,
                                                              self.model)
        self.torchModelProcess.load_latest_optimizer(self.train_task_config.latest_optimizer_path,
                                                     self.optimizer)

    def train(self, train_path, val_path):
        dataloader = get_sr_train_dataloader(train_path, self.train_task_config)
        self.total_images = len(dataloader)
        self.load_latest_param(self.train_task_config.latest_weights_path)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_images)
        lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.lr_scheduler_config)

        self.train_task_config.save_config()
        self.timer.tic()
        self.set_model_train()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            # self.optimizer = torchOptimizer.adjust_optimizer(epoch, lr)
            self.optimizer.zero_grad()
            for idx, (images, labels) in enumerate(dataloader):
                current_idx = epoch * self.total_images + idx
                lr = lr_scheduler.get_lr(epoch, current_idx)
                lr_scheduler.adjust_learning_rate(self.optimizer, lr)
                loss = self.compute_backward(images, labels, idx)
                self.update_logger(idx, self.total_images, epoch, loss)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss = self.compute_loss(output_list, targets)
        loss.backward()

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss

    def compute_loss(self, output_list, targets, loss_type=0):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], targets)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                loss += self.model.lossList[k](output_list[k], targets)
        else:
            print("compute loss error")
        return loss

    def update_logger(self, index, total, epoch, loss):
        loss_value = loss.data.cpu().squeeze()
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        self.train_logger.train_log(step, loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)

        print('Epoch: {}[{}/{}]\t Loss: {}\t Rate: {} \t Time: {}\t'.format(epoch,
                                                                            index,
                                                                            total,
                                                                            '%.7f' % loss_value,
                                                                            '%.7f' % lr,
                                                                            '%.5f' % self.timer.toc(True)))

    def save_train_model(self, epoch):
        self.train_logger.epoch_train_log(epoch)
        if self.train_task_config.is_save_epoch_model:
            save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                           "seg_model_epoch_%d.pt" % epoch)
        else:
            save_model_path = self.train_task_config.latest_weights_path
        self.torchModelProcess.save_latest_model(epoch, self.best_score,
                                                 self.model, save_model_path)
        self.torchModelProcess.save_optimizer_state(epoch, self.optimizer,
                                                    self.train_task_config.latest_weights_path)
        return save_model_path

    def set_model_train(self):
        self.model.train()
        self.freeze_process.freeze_bn(self.model,
                                      self.train_task_config.freeze_bn_layer_name,
                                      self.train_task_config.freeze_bn_type)

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.sr_test.load_weights(save_model_path)
            score, average_loss = self.sr_test.test(val_path)
            self.sr_test.save_test_value(epoch, score)

            self.train_logger.eval_log("val epoch loss", epoch, average_loss)
            print("Val epoch loss: {}".format(average_loss))
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(score,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            print("no test!")


if __name__ == '__main__':
    pass
