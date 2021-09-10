#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.sr.super_resolution import SuperResolution
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.SuperResolution_Task)
class SuperResolutionTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.SuperResolution_Task)
        self.sr_inference = SuperResolution(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for i, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.sr_inference.infer(batch_data)
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.eval(loss_value)
            self.metirc_loss(i, loss_value)

        score = self.evaluation.get_score()
        self.save_test_value(epoch, score)
        print("Val epoch loss: {:.7f}".format(self.epoch_loss_average.avg))
        return score, self.epoch_loss_average.avg

    def save_test_value(self, epoch, score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | psnr: {:.5f} | ".format(epoch, score))
            file.write("\n")

