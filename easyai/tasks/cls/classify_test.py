#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.cls.classify import Classify
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.Classify_Task)
class ClassifyTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.Classify_Task)
        self.inference = Classify(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        top1, loss_value = self.test(epoch)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        print("top1: {.5f}".format(top1))

    def test(self, epoch=0):
        for index, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(batch_data)
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.torch_eval(prediction.data,
                                       batch_data['label'].to(prediction.device))
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        top1 = self.evaluation.get_top1()
        self.save_test_value(epoch)
        EasyLogger.info("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return top1, self.epoch_loss_average.avg

    def save_test_value(self, epoch):
        # Write epoch results
        k = self.evaluation.get_k()
        if max(k) > 1:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec{}: {:.3f} | prec{}: {:.3f}\n".format(epoch,
                                                                                  k[0],
                                                                                  k[1],
                                                                                  self.evaluation.get_top1(),
                                                                                  self.evaluation.get_topK()))
        else:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec1: {:.3f}\n".format(epoch, self.evaluation.get_top1()))

