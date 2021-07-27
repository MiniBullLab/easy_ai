#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.seg.segment import Segmentation
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Segment_Task)
class SegmentionTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Segment_Task)
        self.inference = Segmentation(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.output_process = SegmentResultProcess(self.test_task_config.data['image_size'],
                                                   self.test_task_config.data['resize_type'],
                                                   self.test_task_config.post_process)
        self.evaluation_args = {"type": EvaluationName.SegmentionMetric,
                                'num_class': len(self.test_task_config.segment_name)}
        self.evaluation = self.evaluation_factory.get_evaluation(self.evaluation_args)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        score, loss_value = self.test(epoch)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        print("Mean IoU: {.5f}".format(score))

    def test(self, epoch=0):
        for i, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(batch_data)
            result, _ = self.output_process.post_process(prediction)
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.numpy_eval(result, batch_data['label'][0].data.cpu().numpy())
            self.metirc_loss(i, loss_value)
            self.print_test_info(i, loss_value)

        score, class_score = self.evaluation.get_score()
        self.save_test_value(epoch, score, class_score)
        EasyLogger.info("Val epoch loss: {:.7f}".format(self.epoch_loss_average.avg))
        return score['Mean IoU'], self.epoch_loss_average.avg

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        targets = targets.to(self.device)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                output, target = self.output_process.output_feature_map_resize(output_list[0], targets)
                loss = self.model.lossList[0](output, target)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.lossList[0](output_list, targets)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    output, target = self.output_process.output_feature_map_resize(output_list[k], targets)
                    loss += self.model.lossList[k](output, target)
            else:
                EasyLogger.error("compute loss error")
        return loss.item()

    def save_test_value(self, epoch, score, class_score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU']))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_class[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

