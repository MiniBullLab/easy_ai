#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.key_point_accuracy import KeyPointAccuracy
from easyai.tasks.keypoint2d.keypoint2d import KeyPoint2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.KeyPoint2d_Task)
class KeyPoint2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.KeyPoint2d_Task)
        self.inference = KeyPoint2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(5e-3)
        self.evaluator = KeyPointAccuracy(self.test_task_config.points_count,
                                          self.test_task_config.points_class)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        self.evaluator.reset()
        self.start_test()
        for i, (images, labels) in enumerate(self.dataloader):
            print('%g/%g' % (i + 1, self.total_batch_image), end=' ')
            prediction = self.inference.infer(images)
            result, _ = self.inference.result_process.post_process(prediction, self.conf_threshold)
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
