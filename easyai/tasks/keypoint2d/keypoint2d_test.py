#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.keypoint2d.keypoint2d import KeyPoint2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.KeyPoint2d_Task)
class KeyPoint2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.KeyPoint2d_Task)
        self.inference = KeyPoint2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(5e-3)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for i, batch_data in enumerate(self.dataloader):
            prediction = self.inference.infer(batch_data)
            result, _ = self.inference.result_process.post_process(prediction,
                                                                   self.conf_threshold)
            self.evaluation.eval(result, batch_data['label'][0].data.cpu().numpy())
            self.print_test_info(i)
        accuracy, _ = self.evaluation.get_accuracy()
        self.save_test_value(epoch, accuracy)
        return accuracy

    def save_test_value(self, epoch, accuracy):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | accuracy: {:.3f}".format(epoch, accuracy))
            file.write("\n")
