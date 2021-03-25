#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.key_point_accuracy import KeyPointAccuracy
from easyai.data_loader.keypoint2d.keypoint2d_dataloader import get_key_points2d_val_dataloader
from easyai.tasks.keypoint2d.keypoint2d import KeyPoint2d
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.KeyPoint2d_Task)
class KeyPoint2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.KeyPoint2d_Task)
        self.inference = KeyPoint2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluator = KeyPointAccuracy(self.test_task_config.points_count,
                                          self.test_task_config.points_class)

        self.conf_threshold = 5e-3

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        dataloader = get_key_points2d_val_dataloader(val_path,
                                                     self.test_task_config)

        self.timer.tic()
        self.evaluator.reset()
        for i, (images, labels) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            prediction = self.inference.infer(images)
            result, _ = self.inference.postprocess(prediction, self.conf_threshold)
            labels = labels[0].data.cpu().numpy()
            self.evaluator.eval(result, labels)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))
        accuracy, _ = self.evaluator.get_accuracy()
        self.save_test_value(epoch, accuracy)
        return accuracy

    def save_test_value(self, epoch, accuracy):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | accuracy: {:.3f}".format(epoch, accuracy))
            file.write("\n")
