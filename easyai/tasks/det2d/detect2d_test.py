#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.detection_mAP import DetectionMeanAp
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Detect2d_Task)
class Detection2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Detect2d_Task)
        self.inference = Detection2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.inference.result_process.set_threshold(5e-3)
        self.evaluation = DetectionMeanAp(self.test_task_config.detect2d_class)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        os.system('rm -rf ' + self.test_task_config.save_result_dir)
        os.makedirs(self.test_task_config.save_result_dir, exist_ok=True)
        for i, (image_path, src_size, input_image) in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(input_image)
            detection_objects = self.inference.result_process.post_process(prediction,
                                                                           src_size.numpy()[0])
            self.print_test_info(i)
            self.inference.save_result(image_path[0], detection_objects, 1)

        mAP, aps = self.evaluation.eval(self.test_task_config.save_result_dir,
                                        self.val_path)
        self.save_test_value(epoch, mAP, aps)
        return mAP

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

    def save_test_value(self, epoch, mAP, aps):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect2d_class[i] + ": {:.3f} ".format(ap))
            file.write("\n")

