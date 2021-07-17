#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.cls.classify import Classify
from easyai.name_manager.evaluation_name import EvaluationName
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
        self.evaluation_args = {"type": EvaluationName.ClassifyAccuracy,
                                'top_k': (1,)}
        self.evaluation = self.evaluation_factory.get_evaluation(self.evaluation_args)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for index, batch_data in enumerate(self.dataloader):
            prediction, output_list = self.inference.infer(batch_data['image'])
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.torch_eval(prediction.data,
                                       batch_data['label'].to(prediction.device))
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        top1 = self.evaluation.get_top1()
        self.save_test_value(epoch)
        EasyLogger.info("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        # print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return top1, self.epoch_loss_average.avg

    def save_test_value(self, epoch):
        # Write epoch results
        top_k = self.evaluation_args['top_k']
        if max(top_k) > 1:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec{}: {:.3f} | prec{}: {:.3f}\n".format(epoch,
                                                                                  top_k[0],
                                                                                  top_k[1],
                                                                                  self.evaluation.get_top1(),
                                                                                  self.evaluation.get_topK()))
        else:
            with open(self.test_task_config.evaluation_result_path, 'a') as file:
                file.write("Epoch: {} | prec1: {:.3f}\n".format(epoch, self.evaluation.get_top1()))

