#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.seg.segment_dataloader import get_segment_val_dataloader
from easyai.tasks.seg.segment import Segmentation
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.evaluation.segmen_metric import SegmentionMetric
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Segment_Task)
class SegmentionTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Segment_Task)
        self.segment_inference = Segmentation(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.output_process = SegmentResultProcess(self.test_task_config.image_size,
                                                   self.test_task_config.resize_type)

        self.metric = SegmentionMetric(len(self.test_task_config.segment_class))
        self.threshold = 0.5  # binary class threshold

    def load_weights(self, weights_path):
        self.segment_inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        dataloader = get_segment_val_dataloader(val_path, self.test_task_config)
        print("Eval data num: {}".format(len(dataloader)))
        self.timer.tic()
        self.metric.reset()
        self.epoch_loss_average.reset()
        for i, (images, segment_targets) in enumerate(dataloader):
            prediction, output_list = self.segment_inference.infer(images)
            result = self.output_process.get_segmentation_result(prediction,
                                                                 self.threshold)
            loss = self.compute_loss(output_list, segment_targets)
            gt = segment_targets[0].data.cpu().numpy()
            self.metric.numpy_eval(result, gt)
            self.metirc_loss(i, loss)

        score, class_score = self.metric.get_score()
        self.save_test_value(epoch, score, class_score)
        print("Val epoch loss: {:.7f}".format(self.epoch_loss_average.avg))
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
                print("compute loss error")
        return loss

    def metirc_loss(self, step, loss):
        loss_value = loss.item()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {:.7f} | Time: {:.5f}".format(step,
                                                                loss_value,
                                                                self.timer.toc(True)))

    def save_test_value(self, epoch, score, class_score):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | mIoU: {:.3f} | ".format(epoch, score['Mean IoU : \t']))
            for i, iou in class_score.items():
                file.write(self.test_task_config.segment_class[i][0] + ": {:.3f} ".format(iou))
            file.write("\n")

