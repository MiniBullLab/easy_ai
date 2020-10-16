#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.calculate_mAp import CalculateMeanAp
from easyai.data_loader.det2d.det2d_val_dataloader import get_detection_val_dataloader
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Detect2d_Task)
class Detection2dTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Detect2d_Task)
        self.detect_inference = Detection2d(cfg_path, gpu_id, config_path)
        self.evaluator = CalculateMeanAp(self.test_task_config.detect2d_class)
        self.threshold_det = 5e-3

    def load_weights(self, weights_path):
        self.detect_inference.load_weights(weights_path)

    def test(self, val_path):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)

        dataloader = get_detection_val_dataloader(val_path, self.test_task_config)
        self.timer.tic()
        for i, (image_path, src_image, input_image) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            self.detect_inference.set_src_size(src_image.numpy()[0])

            result = self.detect_inference.infer(input_image, self.threshold_det)
            detection_objects = self.detect_inference.postprocess(result)

            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))
            self.detect_inference.save_result(image_path[0], detection_objects, 1)

        mAP, aps = self.evaluator.eval(self.test_task_config.save_result_dir, val_path)
        return mAP, aps

    def save_test_value(self, epoch, mAP, aps):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_name, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect2d_class[i] + ": {:.3f} ".format(ap))
            file.write("\n")
