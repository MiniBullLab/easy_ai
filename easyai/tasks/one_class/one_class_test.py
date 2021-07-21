#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.one_class.one_class import OneClass
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_TEST_TASK.register_module(TaskName.OneClass)
class OneClassTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.OneClass)
        self.inference = OneClass(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluation_args = {"type": EvaluationName.OneClassROC,
                                'save_dir': self.test_task_config.root_save_dir}
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
            prediction, output_list = self.inference.infer(batch_data)
            loss_value = self.compute_loss(output_list, batch_data)
            self.evaluation.eval(prediction, batch_data['label'].detach().numpy())
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        roc_auc = self.evaluation.get_score()
        self.save_test_value(epoch, roc_auc)
        EasyLogger.info("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        # print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return roc_auc, self.epoch_loss_average.avg

    def compute_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.g_loss_list[0](output_list[0], batch_data)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.g_loss_list[0](output_list, batch_data)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.g_loss_list[k](output_list[k], batch_data)
            else:
                print("compute loss error")
        return loss.item()

    def save_test_value(self, epoch, roc_auc):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | AUC: {:.3f} \n".format(epoch, roc_auc))
