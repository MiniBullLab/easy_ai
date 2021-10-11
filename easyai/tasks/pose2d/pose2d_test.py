#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.pose2d.pose2d import Pose2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Pose2d_Task)
class Pose2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Pose2d_Task)
        self.inference = Pose2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(1e-5)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for index, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(batch_data)
            result, _ = self.inference.result_process.post_process(prediction, (0, 0))
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.eval(result, batch_data['label'].detach().numpy())
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        average_socre = self.evaluation.get_score()
        self.save_test_value(epoch, average_socre)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return average_socre, self.epoch_loss_average.avg

    def save_test_value(self, epoch, score):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | prec: {:.3f}\n".format(epoch, score))


