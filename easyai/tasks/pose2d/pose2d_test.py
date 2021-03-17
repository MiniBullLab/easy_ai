#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.calculate_mAp import CalculateMeanAp
from easyai.data_loader.det2d.det2d_val_dataloader import get_detection_val_dataloader
from easyai.tasks.pose2d.pose2d import Pose2d
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Pose2d_Task)
class Pose2dTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Pose2d_Task)
        self.pose2d_inference = Pose2d(cfg_path, gpu_id, config_path)
        self.evaluator = CalculateMeanAp(self.test_task_config.detect2d_class)
        self.threshold_det = 5e-3

    def load_weights(self, weights_path):
        self.pose2d_inference.load_weights(weights_path)

    def test(self, val_path):
        if self.task_config.trian_det:
            print("test det")
            self.test_det(val_path)
        if self.task_config.trian_pose:
            print("test pose")
            self.test_pose(val_path)

    def test_det(self, val_path):
        os.system('rm -rf ' + self.test_task_config.det_config.save_result_dir)
        os.makedirs(self.test_task_config.det_config.save_result_dir, exist_ok=True)

        dataloader = get_detection_val_dataloader(val_path, self.test_task_config)
        self.timer.tic()
        for i, (image_path, src_size, input_image) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')

            prediction = self.pose2d_inference.infer(input_image, 0)

            detection_objects = self.pose2d_inference.det_result_process.postprocess(prediction,
                                                                                     src_size.numpy()[0],
                                                                                     self.threshold_det)

            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc(True)))
            self.pose2d_inference.save_result(image_path[0], detection_objects, 1)

        mAP, aps = self.evaluator.eval(self.test_task_config.save_result_dir, val_path)
        self.evaluator.print_evaluation(aps)
        return mAP, aps

    def test_pose(self, val_path):
        pass

    def save_det_value(self, epoch, mAP, aps):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            # file.write('%11.3g' * 2 % (mAP, aps[0]) + '\n')
            file.write("Epoch: {} | mAP: {:.3f} | ".format(epoch, mAP))
            for i, ap in enumerate(aps):
                file.write(self.test_task_config.detect2d_class[i] + ": {:.3f} ".format(ap))
            file.write("\n")

    def metirc_loss(self, step, loss):
        loss_value = loss.item()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {:.7f} | Time: {:.5f}".format(step,
                                                                loss_value,
                                                                self.timer.toc(True)))