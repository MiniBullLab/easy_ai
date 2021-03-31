#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.sr.super_resolution_dataloader import get_sr_val_dataloader
from easyai.tasks.sr.super_resolution import SuperResolution
from easyai.evaluation.super_resolution_psnr import SuperResolutionPSNR
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.SuperResolution_Task)
class SuperResolutionTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.SuperResolution_Task)
        self.sr_inference = SuperResolution(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()

        self.evalution = SuperResolutionPSNR()

    def load_weights(self, weights_path):
        self.sr_inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        dataloader = get_sr_val_dataloader(val_path, self.test_task_config)
        self.total_batch_image = len(dataloader)
        self.evalution.reset()
        self.start_test()
        for i, (images, sr_targets) in enumerate(dataloader):
            prediction, output_list = self.sr_inference.infer(images)
            loss = self.compute_loss(output_list, sr_targets)
            self.compute_metric(loss)
            self.metirc_loss(i, loss)

        score = self.evalution.get_score()
        self.save_test_value(epoch, score)
        print("Val epoch loss: {:.7f}".format(self.epoch_loss_average.avg))
        return score, self.epoch_loss_average.avg

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

    def metirc_loss(self, step, loss):
        loss_value = loss.data.cpu().squeeze()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {} | Time: {.7f}".format(step,
                                                           loss_value,
                                                           self.timer.toc(True)))

    def save_test_value(self, epoch, score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | psnr: {:.5f} | ".format(epoch, score))
            file.write("\n")

