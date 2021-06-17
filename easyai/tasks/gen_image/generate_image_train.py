#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.gan_train import GanTrain
from easyai.tasks.gen_image.generate_image import GenerateImage
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.GenerateImage)
class GenerateImageTrain(GanTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.GenerateImage)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'],
                             image_size=self.train_task_config.data['image_size'])
        self.set_model(gpu_id=gpu_id)

        self.gen_test = GenerateImage(model_name, gpu_id, self.train_task_config)

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score \
                = self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)
        self.build_optimizer()

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
        for i, (images, targets) in enumerate(dataloader):
            current_iter = epoch * self.total_batch_image + i
            d_lr = d_lr_scheduler.get_lr(epoch, current_iter)
            for optimizer in self.d_optimizer_list:
                d_lr_scheduler.adjust_learning_rate(optimizer, d_lr)
            g_lr = g_lr_scheduler.get_lr(epoch, current_iter)
            for optimizer in self.g_optimizer_list:
                g_lr_scheduler.adjust_learning_rate(optimizer, g_lr)
            loss_values = self.compute_backward(images, targets, i)
            self.update_logger(i, self.total_images, epoch, loss_values)

    def compute_backward(self, input_datas, targets, step_index):
        # Compute loss, compute gradient, update parameters
        d_loss_values = None
        g_loss_values = None
        if step_index == 0 or (step_index % self.train_task_config.g_skip_batch_backward == 0):
            g_loss_values = self.generator_backward(input_datas, targets)
        if step_index == 0 or (step_index % self.train_task_config.d_skip_batch_backward == 0):
            d_loss_values = self.discriminator_backward(input_datas, targets)
        return d_loss_values, g_loss_values

    def generator_backward(self, input_datas, targets):
        fake_images = self.model.generator_input_data(input_datas, 0)
        output_list = self.model(fake_images.to(self.device),
                                 net_type=1)
        loss = self.compute_g_loss(output_list, targets)
        for index, optimizer in enumerate(self.g_optimizer_list):
            optimizer.zero_grad()
            loss[index].backward()
            optimizer.step()
        return loss

    def discriminator_backward(self, input_datas, targets):
        fake_images = self.model.generator_input_data(input_datas, 0)
        real_images = self.model.generator_input_data(input_datas, 1)
        output_list = self.model(fake_images.to(self.device),
                                 real_images.to(self.device),
                                 net_type=2)
        loss = self.compute_d_loss(output_list, targets)
        for index, optimizer in enumerate(self.d_optimizer_list):
            optimizer.zero_grad()
            loss[index].backward()
            optimizer.step()
        return loss

    def compute_g_loss(self, output_list, targets):
        loss = []
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            result = self.model.g_loss_list[0](output_list[0], targets)
            loss.append(result)
        elif loss_count == 1 and output_count > 1:
            result = self.model.g_loss_list[0](output_list, targets)
            loss.append(result)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):
                result = self.model.g_loss_list[k](output_list[k], targets)
                loss.append(result)
        else:
            print("compute generator loss error")
        return loss

    def compute_d_loss(self, output_list, targets):
        loss = []
        loss_count = len(self.model.d_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            result = self.model.d_loss_list[0](output_list[0], targets)
            loss.append(result)
        elif loss_count == 1 and output_count > 1:
            result = self.model.d_loss_list[0](output_list, targets)
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
        if epoch % 5 == 0:
            self.gen_test.load_weights(save_model_path)
            self.gen_test.process(None, 0, False)
