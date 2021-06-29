#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.key_point_accuracy import KeyPointAccuracy
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
        self.evaluation = KeyPointAccuracy(self.test_task_config.points_count,
                                           self.test_task_config.points_class)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for i, (images, labels) in enumerate(self.dataloader):
            prediction = self.inference.infer(images)
            result, _ = self.inference.result_process.post_process(prediction, self.conf_threshold)
            labels = labels[0].data.cpu().numpy()
            self.evaluation.eval(result, labels)
            self.print_test_info(i)
        accuracy, _ = self.evaluation.get_accuracy()
        self.save_test_value(epoch, accuracy)
        return accuracy

    def save_test_value(self, epoch, accuracy):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | accuracy: {:.3f}".format(epoch, accuracy))
            file.write("\n")
