#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.polygon2d.polygon2d import Polygon2d
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Polygon2d_Task)
class Polygon2dTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Polygon2d_Task)
        self.inference = Polygon2d(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        score, loss_value = self.test(epoch)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        print("Mean IoU: {:.5f}".format(score))

    def test(self, epoch=0):
        for i, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(batch_data)
            result, _ = self.inference.result_process.post_process(prediction)
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.numpy_eval(result, batch_data['label'][0].data.cpu().numpy())
            self.metirc_loss(i, loss_value)
            self.print_test_info(i, loss_value)

        score, class_score = self.evaluation.get_score()
        self.save_test_value(epoch, score, class_score)
        EasyLogger.info("Val epoch loss: {:.7f}".format(self.epoch_loss_average.avg))
        return score['Mean IoU'], self.epoch_loss_average.avg

    def save_test_value(self, epoch, score, class_score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU']))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_class[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

