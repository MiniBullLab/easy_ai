#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.seg.segment import Segmentation
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.evaluation.segmen_metric import SegmentionMetric
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Segment_Task)
class SegmentionTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Segment_Task)
        self.segment_inference = Segmentation(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.output_process = SegmentResultProcess(self.test_task_config.image_size,
                                                   self.test_task_config.resize_type,
                                                   self.task_config.post_process)

        self.evaluation = SegmentionMetric(len(self.test_task_config.segment_class))

    def load_weights(self, weights_path):
        self.segment_inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for i, (images, segment_targets) in enumerate(self.dataloader):
            prediction, output_list = self.segment_inference.infer(images)
            result, _ = self.output_process.post_process(prediction)
            loss_value = self.compute_loss(output_list, segment_targets)
            gt = segment_targets[0].data.cpu().numpy()
            self.evaluation.numpy_eval(result, gt)
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

