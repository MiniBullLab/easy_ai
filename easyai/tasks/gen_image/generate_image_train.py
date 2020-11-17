#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.helper.average_meter import AverageMeter
from easyai.data_loader.gan.gan_dataloader import get_gan_train_dataloader
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.tasks.utility.base_train import BaseTrain
from easyai.base_name.task_name import TaskName


class GenerateImageTrain(BaseTrain):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.GenerateImage)

        self.model_args['image_size'] = len(self.train_task_config.image_size)
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)

        self.d_loss_average = AverageMeter()
        self.g_loss_average = AverageMeter()

        self.d_optimizer_list = []
        self.g_optimizer_list = []
        self.total_images = 0
        self.start_epoch = 0
        self.best_score = 0

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score \
                = self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)

        self.freeze_process.freeze_block(self.model,
                                         self.train_task_config.freeze_layer_name,
                                         self.train_task_config.freeze_layer_type)

        self.load_optimize_param()

    def load_optimize_param(self):
        d_optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                       self.train_task_config.d_optimizer_config)
        g_optimizer_args = self.optimizer_process.get_optimizer_config(self.start_epoch,
                                                                       self.train_task_config.g_optimizer_config)
        for d_model in self.model.d_model_list:
            optimizer = self.optimizer_process.get_optimizer(d_optimizer_args,
                                                             d_model)
            self.d_optimizer_list.append(optimizer)

        for g_model in self.model.g_model_list:
            optimizer = self.optimizer_process.get_optimizer(g_optimizer_args,
                                                             g_model)
            self.d_optimizer_list.append(optimizer)

    def train(self, train_path, val_path):
        dataloader = get_gan_train_dataloader(train_path, self.train_task_config)
        self.total_images = len(dataloader)

        lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                        self.train_task_config.max_epochs,
                                        self.total_images)
        d_lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.d_lr_scheduler_config)
        g_lr_scheduler = lr_factory.get_lr_scheduler(self.train_task_config.g_lr_scheduler_config)

        self.load_latest_param(self.train_task_config.latest_weights_path)

        self.train_task_config.save_config()
        self.timer.tic()
        self.model.train()
        self.d_loss_average.reset()
        self.g_loss_average.reset()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            for i, (images, targets) in enumerate(dataloader):
                current_iter = epoch * self.total_images + i
                d_lr = d_lr_scheduler.get_lr(epoch, current_iter)
                for optimizer in self.d_optimizer_list:
                    d_lr_scheduler.adjust_learning_rate(optimizer, d_lr)
                g_lr = g_lr_scheduler.get_lr(epoch, current_iter)
                for optimizer in self.g_optimizer_list:
                    g_lr_scheduler.adjust_learning_rate(optimizer, g_lr)
                loss_values = self.compute_backward(images, targets, i)
                self.update_logger(i, self.total_images, epoch, loss_values)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        d_loss_values = self.discriminator_backward(input_datas, targets)
        g_loss_values = self.generator_backward(input_datas, targets)
        return d_loss_values, g_loss_values

    def discriminator_backward(self, input_datas, targets):
        real_images = self.model.generator_input_data(input_datas, 0)
        fake_images = self.model.generator_input_data(input_datas, 1)
        output_list = self.model(fake_images.to(self.device),
                                 real_images.to(self.device),
                                 net_type=1)
        loss = self.compute_loss(output_list, targets, 0)
        for index, optimizer in enumerate(self.d_optimizer_list):
            optimizer.zero_grad()
            loss[index].backward()
            optimizer.step()
        return loss

    def generator_backward(self, input_datas, targets):
        fake_images = self.model.generator_input_data(input_datas, 1)
        output_list = self.model(fake_images.to(self.device),
                                 net_type=2)
        loss = self.compute_loss(output_list, targets, 1)
        for index, optimizer in enumerate(self.g_optimizer_list):
            optimizer.zero_grad()
            loss[index].backward()
            optimizer.step()
        return loss

    def compute_loss(self, output_list, targets, loss_type=0):
        loss = []
        if loss_type == 0:
            loss = self.compute_d_loss(output_list, targets)
        elif loss_type == 1:
            loss = self.compute_g_loss(output_list, targets)
        else:
            print("compute loss error")
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

    def update_logger(self, index, total, epoch, loss_values):
        step = epoch * total + index
        d_loss_values, g_loss_values = loss_values
        all_d_loss_value = 0
        all_g_loss_value = 0
        for index, d_loss in enumerate(d_loss_values):
            tag = "d_loss_%d" % index
            all_d_loss_value += d_loss.item()
            self.train_logger.add_scalar(tag, d_loss.item(), step)
        for index, g_loss in enumerate(g_loss_values):
            tag = "g_loss_%d" % index
            all_g_loss_value += g_loss.item()
            self.train_logger.add_scalar(tag, g_loss.item(), step)

        for index, optimizer in enumerate(self.d_optimizer_list):
            tag = "d_lr_%d" % index
            d_lr = optimizer.param_groups[0]['lr']
            self.train_logger.add_scalar(tag, d_lr, step)
        for index, optimizer in enumerate(self.g_optimizer_list):
            tag = "g_lr_%d" % index
            g_lr = optimizer.param_groups[0]['lr']
            self.train_logger.add_scalar(tag, g_lr, step)

        self.d_loss_average.update(all_d_loss_value)
        self.g_loss_average.update(all_g_loss_value)

        print('Epoch: {} \t Time: {:.5f}\t'.format(epoch, self.timer.toc(True)))
        print('Epoch: {}[{}/{}]\t  D Loss: {:.7f}\t G Loss: {:.7f} \t'.format(epoch, index, total,
                                                                              self.d_loss_average.avg,
                                                                              self.g_loss_average.avg))

    def save_train_model(self, epoch):
        self.train_logger.add_scalar("train epoch d loss",
                                     self.d_loss_average.avg, epoch)
        self.train_logger.add_scalar("train epoch g loss",
                                     self.g_loss_average.avg, epoch)
        self.d_loss_average.reset()
        self.g_loss_average.reset()

        if self.train_task_config.is_save_epoch_model:
            save_model_path = os.path.join(self.train_task_config.snapshot_path,
                                           "generate_image_epoch_%d.pt" % epoch)
        else:
            save_model_path = self.train_task_config.latest_weights_path
        self.torchModelProcess.save_latest_model(epoch, 0, self.model, save_model_path)

        return save_model_path

    def test(self, val_path, epoch, save_model_path):
        pass
