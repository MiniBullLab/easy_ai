#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.tasks.utility.common_train import CommonTrain
from easyai.tasks.multi_task.det2d_seg_task_test import Det2dSegTaskTest
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TRAIN_TASK


@REGISTERED_TRAIN_TASK.register_module(TaskName.Det2d_Seg_Task)
class Det2dSegTaskTrain(CommonTrain):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Det2d_Seg_Task)
        self.set_model_param(data_channel=self.train_task_config.data['data_channel'])
        self.set_model(gpu_id=gpu_id)
        self.multi_task_test = Det2dSegTaskTest(model_name, gpu_id, self.train_task_config)

        self.avg_loss = -1
        self.best_mAP = 0
        self.bestmIoU = 0

    def load_latest_param(self, latest_weights_path):
        if latest_weights_path and os.path.exists(latest_weights_path):
            self.start_epoch, self.best_score = \
                self.torchModelProcess.load_latest_model(latest_weights_path, self.model)

        self.model = self.torchModelProcess.model_train_init(self.model)
        self.build_optimizer()

    def train(self, train_path, val_path):
        self.create_dataloader(train_path)
        self.build_lr_scheduler()
        self.load_latest_param(self.train_task_config.latest_weights_path)
        self.start_train()
        for epoch in range(self.start_epoch, self.train_task_config.max_epochs):
            self.optimizer.zero_grad()
            for i, (images, detects, segments) in enumerate(self.dataloader):
                current_iter = epoch * self.total_batch_image + i
                lr = self.lr_scheduler.get_lr(epoch, current_iter)
                self.lr_scheduler.adjust_learning_rate(self.optimizer, lr)
                if sum([len(x) for x in detects]) < 1:  # if no targets continue
                    continue
                # the order if output in my cfg is segment, detct1, detect2, detect3
                targets = [segments, detects, detects, detects]
                loss, loss_list = self.compute_backward(images, targets, i)
                self.update_logger(i, self.total_images, epoch, loss_list)

            save_model_path = self.save_train_model(epoch)
            self.test(val_path, epoch, save_model_path)

    def compute_backward(self, input_datas, targets, setp_index):
        # Compute loss, compute gradient, update parameters
        output_list = self.model(input_datas.to(self.device))
        loss, loss_list = self.compute_loss(output_list, targets)

        self.loss_backward(loss)

        # accumulate gradient for x batches before optimizing
        if ((setp_index + 1) % self.train_task_config.accumulated_batches == 0) \
                or (setp_index == self.total_images - 1):
            self.clip_grad()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, loss_list

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_list = []
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets[0] = targets[0].to(self.device)
        if loss_count == 1 and output_count == 1:
            loss = self.model.lossList[0](output_list[0], targets[0])
        elif loss_count == 1 and output_count > 1:
            loss = self.model.lossList[0](output_list, targets)
        elif loss_count > 1 and loss_count == output_count:
            for k in range(0, loss_count):  # loss_count
                loss_list.append(self.model.lossList[k](output_list[k], targets[k]))
                loss = sum(loss_list)
        else:
            print("compute loss error")
        return loss, loss_list

    def update_logger(self, index, total, epoch, loss_list):
        step = epoch * total + index
        lr = self.optimizer.param_groups[0]['lr']
        seg_loss = loss_list[0]
        seg_loss_value = seg_loss.data.cpu().squeeze()
        det_loss = sum(loss_list[1:-1])
        det_loss_value = det_loss.data.cpu().squeeze()

        if self.avg_loss < 0:
            self.avg_loss = (det_loss.cpu().detach().numpy() / self.train_task_config.train_batch_size)
        self.avg_loss = 0.9 * (det_loss.cpu().detach().numpy() / self.train_task_config.train_batch_size) \
                        + 0.1 * self.avg_loss

        self.train_logger.loss_log(step, seg_loss_value, self.train_task_config.display)
        self.train_logger.loss_log(step, det_loss_value, self.train_task_config.display)
        self.train_logger.lr_log(step, lr, self.train_task_config.display)
        print('Epoch: {}[{}/{}]\t Loss_seg: {:.7f}\t '
              'Loss_det: {:.7f}\t Rate: {:.7f} \t Time: {:.5f}\t'.format(epoch, index, total,
                                                                         seg_loss.item(),
                                                                         self.avg_loss,
                                                                         lr,
                                                                         self.timer.toc(True)))

    def test(self, val_path, epoch, save_model_path):
        if val_path is not None and os.path.exists(val_path):
            self.multi_task_test.load_weights(save_model_path)
            mAP, score = self.multi_task_test.test(val_path, epoch)
            # wrong !!! how to use best_iou
            # save best model
            self.best_score = self.torchModelProcess.save_best_model(mAP, save_model_path,
                                                                     self.train_task_config.best_weights_path)
        else:
            print("no test!")
