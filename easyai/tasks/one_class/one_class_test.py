#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_test import BaseTest
from easyai.tasks.one_class.one_class import OneClass
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

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        for index, batch_data in enumerate(self.dataloader):
            prediction, model_output = self.inference.infer(batch_data)
            _, score = self.inference.result_process.post_process(prediction)
            loss_value = self.compute_loss(model_output, batch_data)
            self.evaluation.eval(score, batch_data['label'].detach().numpy())
            self.metirc_loss(index, loss_value)
            self.print_test_info(index, loss_value)
        roc_auc = self.evaluation.get_score()
        self.save_test_value(epoch, roc_auc)
        EasyLogger.info("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        # print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return roc_auc, self.epoch_loss_average.avg

    def save_test_value(self, epoch, roc_auc):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | AUC: {:.3f} \n".format(epoch, roc_auc))
