#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.pose2d_accuracy import Pose2dAccuracy
from easyai.data_loader.pose2d.pose2d_dataloader import get_poes2d_val_dataloader
from easyai.helper.average_meter import AverageMeter
from easyai.tasks.pose2d.pose2d import Pose2d
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Pose2d_Task)
class Pose2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Pose2d_Task)
        self.inference = Pose2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluation = Pose2dAccuracy(self.test_task_config.points_count,
                                         self.test_task_config.image_size)
        self.epoch_loss_average = AverageMeter()

    def load_weights(self, weights_path):
        self.pose2d_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_poes2d_val_dataloader(val_path, self.test_task_config)
        all_count = len(dataloader)
        self.evaluation.reset()
        self.epoch_loss_average.reset()
        self.timer.tic()
        for index, (images, targets) in enumerate(dataloader):
            print('%g/%g' % (index + 1, all_count), end=' ')

            prediction, output_list = self.pose2d_inference.infer(images)
            loss = self.compute_loss(output_list, targets)
            self.evaluation.numpy_eval(prediction, targets.detach().numpy())
            self.metirc_loss(index, loss)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc(True)))
        average_loss = self.epoch_loss_average.avg
        self.print_evaluation()
        return self.evaluation.get_score(), average_loss

    def save_test_value(self, epoch):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | prec: {:.3f}\n".format(epoch, self.evaluation.get_score()))

    def metirc_loss(self, step, loss):
        loss_value = loss.item()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {:.7f} | Time: {:.5f}".format(step,
                                                                loss_value,
                                                                self.timer.toc(True)))

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        with torch.no_grad():
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

    def print_evaluation(self):
        print('prec: {:.3f} \t'.format(self.evaluation.get_score()))
