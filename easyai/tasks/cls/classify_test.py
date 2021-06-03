#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.cls.classify_dataloader import get_classify_val_dataloader
from easyai.tasks.cls.classify import Classify
from easyai.evaluation.classify_accuracy import ClassifyAccuracy
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.Classify_Task)
class ClassifyTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Classify_Task)
        self.inference = Classify(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.topK = (1,)
        self.evaluation = ClassifyAccuracy(top_k=self.topK)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        dataloader = get_classify_val_dataloader(val_path, self.test_task_config)
        self.evaluation.clean_data()
        self.total_batch_image = len(dataloader)
        self.start_test()
        for index, (images, labels) in enumerate(dataloader):
            prediction, output_list = self.inference.infer(images)
            loss_value = self.compute_loss(output_list, labels)
            self.evaluation.torch_eval(prediction.data, labels.to(prediction.device))
            self.metirc_loss(index, loss_value)
        top1 = self.evaluation.get_top1()
        self.save_test_value(epoch)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return top1, self.epoch_loss_average.avg

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
        return loss.item()

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

