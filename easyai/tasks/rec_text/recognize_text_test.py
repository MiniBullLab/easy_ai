#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.evaluation.rec_text_metric import RecognizeTextMetric
from easyai.tasks.rec_text.recognize_text import RecognizeText
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.RecognizeText)
class RecognizeTextTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.RecognizeText)
        self.inference = RecognizeText(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluation = RecognizeTextMetric()

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def process_test(self, val_path, epoch=0):
        self.create_dataloader(val_path)
        if not self.start_test():
            EasyLogger.info("no test!")
            return
        self.test(epoch)

    def test(self, epoch=0):
        try:
            for index, (images, targets) in enumerate(self.dataloader):
                prediction, output_list = self.inference.infer(images)
                result = self.inference.result_process.post_process(prediction)
                loss_value = self.compute_loss(output_list, targets)
                self.evaluation.eval(result, targets['text'])
                self.metirc_loss(index, loss_value)
                self.print_test_info(index, loss_value)
        except Exception as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        average_socre = self.evaluation.get_score()
        self.save_test_value(epoch, average_socre)
        EasyLogger.info("Val epoch({}) loss: {}".format(epoch, self.epoch_loss_average.avg))
        # print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return average_socre['accuracy'], self.epoch_loss_average.avg

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.lossList[0](output_list[0], targets)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.lossList[0](output_list, targets)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.lossList[k](output_list[k], targets)
            else:
                EasyLogger.error("compute loss error")
        return loss.item()

    def save_test_value(self, epoch, score):
        # Write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | acc: {:.3f} edit_distance: {:.3f}\n".format(epoch,
                                                                                score['accuracy'],
                                                                                score['edit_distance']))



