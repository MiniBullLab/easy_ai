#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.cls.classify_dataloader import get_classify_val_dataloader
from easyai.tasks.cls.classify import Classify
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.helper.average_meter import AverageMeter
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Classify_Task)
class ClassifyTest(BaseTest):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(config_path, TaskName.Classify_Task)
        self.classify_inference = Classify(cfg_path, gpu_id, config_path)
        self.model = self.classify_inference.model
        self.device = self.classify_inference.device
        self.topK = (1,)
        self.evaluation = ClassifyAccuracy(top_k=self.topK)
        self.epoch_loss_average = AverageMeter()

    def load_weights(self, weights_path):
        self.classify_inference.load_weights(weights_path)

    def test(self, val_path):
        dataloader = get_classify_val_dataloader(val_path, self.test_task_config)
        self.evaluation.clean_data()
        self.epoch_loss_average.reset()
        for index, (images, labels) in enumerate(dataloader):
            prediction, output_list = self.classify_inference.infer(images)
            loss = self.compute_loss(output_list, labels)
            self.evaluation.torch_eval(prediction.data, labels.to(prediction.device))
            self.metirc_loss(index, loss)
        average_loss = self.epoch_loss_average.avg
        self.print_evaluation()
        return self.evaluation.get_top1(), average_loss

    def save_test_value(self, epoch):
        # Write epoch results
        if max(self.topK) > 1:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec{}: {:.3f} | prec{}: {:.3f}\n".format(epoch,
                                                                                  self.topK[0],
                                                                                  self.topK[1],
                                                                                  self.evaluation.get_top1(),
                                                                                  self.evaluation.get_topK()))
        else:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec1: {:.3f}\n".format(epoch, self.evaluation.get_top1()))

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
                print("compute loss error")
        return loss

    def metirc_loss(self, step, loss):
        loss_value = loss.item()
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {:.7f} | Time: {:.5f}".format(step,
                                                                loss_value,
                                                                self.timer.toc(True)))

    def print_evaluation(self):
        if max(self.topK) > 1:
            print('prec{}: {:.3f} \t prec{}: {:.3f}\t'.format(self.topK[0],
                                                              self.topK[1],
                                                              self.evaluation.get_top1(),
                                                              self.evaluation.get_topK()))
        else:
            print('prec1: {:.3f} \t'.format(self.evaluation.get_top1()))
