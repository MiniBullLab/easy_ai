#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.landmark_accuracy import LandmarkAccuracy
from easyai.tasks.landmark.landmark import Landmark
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Pose2d_Task)
class Pose2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Pose2d_Task)
        self.inference = Landmark(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(1e-5)
        self.evaluation = LandmarkAccuracy(self.test_task_config.points_count)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for index, (images, targets) in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(images)
            result = self.inference.result_process.post_process(prediction, (0, 0))
            loss_value = self.compute_loss(output_list, targets)
            self.evaluation.eval(result, targets.detach().numpy())
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        average_socre = self.evaluation.get_score()
        self.save_test_value(epoch, average_socre)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return average_socre, self.epoch_loss_average.avg

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
                EasyLogger.error("compute loss error")
        return loss.item()

    def save_test_value(self, epoch, socre):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | AUC: {:.3f}\n".format(epoch, socre))
