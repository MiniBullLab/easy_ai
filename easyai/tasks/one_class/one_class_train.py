#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.gan_train import GanTrain
from easyai.tasks.one_class.one_class_test import OneClassTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TRAIN_TASK.register_module(TaskName.OneClass)
class OneClassTrain(GanTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.OneClass)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'],
                             image_size=self.train_task_config.data['image_size'])
        self.set_model(gpu_id=gpu_id, init_type="normal")
        self.one_class_test = OneClassTest(model_name, gpu_id, self.train_task_config)

    def train(self, train_path, val_path):
        self.create_dataloader(train_path)
        self.lr_factory.set_epoch_iteration(self.total_batch_image)
        d_lr_scheduler = self.lr_factory.get_lr_scheduler(self.train_task_config.d_lr_scheduler_config)
        g_lr_scheduler = self.lr_factory.get_lr_scheduler(self.train_task_config.g_lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_path)

        self.start_train()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            self.trian_epoch(epoch, d_lr_scheduler, g_lr_scheduler, self.dataloader)
            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def trian_epoch(self, epoch, d_lr_scheduler, g_lr_scheduler, dataloader):
        for i, batch_data in enumerate(dataloader):
            current_iter = epoch * self.total_batch_image + i
            g_lr = g_lr_scheduler.get_lr(epoch, current_iter)
            for optimizer in self.g_optimizer_list:
                g_lr_scheduler.adjust_learning_rate(optimizer, g_lr)
            d_lr = d_lr_scheduler.get_lr(epoch, current_iter)
            for optimizer in self.d_optimizer_list:
                d_lr_scheduler.adjust_learning_rate(optimizer, d_lr)
            loss_values = self.compute_backward(batch_data, i)
            self.update_logger(i, self.total_batch_image, epoch, loss_values)

    def compute_backward(self, batch_data, step_index):
        # Compute loss, compute gradient, update parameters
        d_loss_values = None
        g_loss_values = None
        g_output_list = self.model(input_datas.to(self.device),
                                   net_type=1)
        d_output_list = self.model(g_output_list[0], g_output_list[2],
                                   net_type=2)
        if step_index == 0 or (step_index % self.train_task_config.g_skip_batch_backward == 0):
            g_loss_values = self.generator_backward(g_output_list, batch_data)
        if step_index == 0 or (step_index % self.train_task_config.d_skip_batch_backward == 0):
            d_loss_values = self.discriminator_backward(d_output_list, batch_data)
        return d_loss_values, g_loss_values

    def generator_backward(self, output_list, batch_data):
        loss = self.compute_g_loss(output_list, batch_data)
        for temp_index, optimizer in enumerate(self.g_optimizer_list):
            optimizer.zero_grad()
            loss[temp_index].backward()
            optimizer.step()
        return loss

    def discriminator_backward(self, output_list, batch_data):
        loss = self.compute_d_loss(output_list, batch_data)
        for temp_index, optimizer in enumerate(self.d_optimizer_list):
            optimizer.zero_grad()
            loss[temp_index].backward()
            optimizer.step()

            if loss[temp_index].item() < 1e-5:
                self.torchModelProcess.init_model(self.model.d_model_list[temp_index], init_type="normal")
        return loss

    def compute_g_loss(self, output_list, batch_data):
        loss = []
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            result = self.model.g_loss_list[0](output_list[0], batch_data)
            self.model.g_loss_list[0].print_loss_info()
            loss.append(result)
        elif loss_count == 1 and output_count > 1:
            result = self.model.g_loss_list[0](output_list, batch_data)
            self.model.g_loss_list[0].print_loss_info()
            loss.append(result)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                result = self.model.g_loss_list[k](output_list[k], batch_data)
                self.model.g_loss_list[k].print_loss_info()
                loss.append(result)
        else:
            print("compute generator loss error")
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
                result = self.model.d_loss_list[k](output_list[k], targets)
                loss.append(result)
        else:
            print("compute discriminator loss error")
        return loss

    def update_logger(self, index, total, epoch, loss_values):
        d_loss_values, g_loss_values = loss_values
        self.update_d_logger(index, total, epoch, d_loss_values)
        self.update_g_logger(index, total, epoch, g_loss_values)
        print('Epoch: {} \t Time: {:.5f}\t'.format(epoch, self.timer.toc(True)))

    def update_d_logger(self, index, total, epoch, d_loss_values):
        if d_loss_values is None:
            return
        step = epoch * total + index
        all_d_loss_value = 0
        for temp_index, d_loss in enumerate(d_loss_values):
            tag = "d_loss_%d" % temp_index
            all_d_loss_value += d_loss.item()
            self.train_logger.add_scalar(tag, d_loss.item(), step)
        for temp_index, optimizer in enumerate(self.d_optimizer_list):
            tag = "d_lr_%d" % temp_index
            d_lr = optimizer.param_groups[0]['lr']
            self.train_logger.add_scalar(tag, d_lr, step)
            print("d Lr:", d_lr)
        self.d_loss_average.update(all_d_loss_value)
        print('Epoch: {}[{}/{}]\t  D Loss: {:.7f}\t'.format(epoch, index, total,
                                                            self.d_loss_average.avg))

    def update_g_logger(self, index, total, epoch, g_loss_values):
        if g_loss_values is None:
            return
        step = epoch * total + index
        all_g_loss_value = 0
        for temp_index, g_loss in enumerate(g_loss_values):
            tag = "g_loss_%d" % temp_index
            all_g_loss_value += g_loss.item()
            self.train_logger.add_scalar(tag, g_loss.item(), step)
        for temp_index, optimizer in enumerate(self.g_optimizer_list):
            tag = "g_lr_%d" % temp_index
            g_lr = optimizer.param_groups[0]['lr']
            self.train_logger.add_scalar(tag, g_lr, step)
            print("g Lr:", g_lr)
        self.g_loss_average.update(all_g_loss_value)
        print('Epoch: {}[{}/{}]\t  G Loss: {:.7f}\t'.format(epoch, index, total,
                                                            self.g_loss_average.avg))

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path) and \
                epoch % 5 == 0:
            if self.test_first:
                self.classify_test.create_dataloader(val_path)
                self.test_first = False
            if not self.classify_test.start_test():
                EasyLogger.info("no test!")
                return
            self.one_class_test.load_weights(save_model_path)
            roc_auc, average_loss = self.one_class_test.test(epoch)

            self.train_logger.epoch_eval_loss_log(epoch, average_loss)
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(roc_auc,
                                                                     save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            print("no test!")
