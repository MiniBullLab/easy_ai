#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_test import BaseTest
from easyai.data_loader.one_class.one_class_dataloader import get_one_class_val_dataloader
from easyai.tasks.one_class.one_class import OneClass
from easyai.evaluation.one_class_roc import OneClassROC
from easyai.config.name_manager import TaskName
from easyai.tasks.utility.registry import REGISTERED_TEST_TASK


@REGISTERED_TEST_TASK.register_module(TaskName.OneClass)
class OneClassTest(BaseTest):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(TaskName.OneClass)
        self.inference = OneClass(model_name, gpu_id, config_path)
        self.set_test_config(self.inference.task_config)
        self.set_model()
        self.evaluation = OneClassROC(self.test_task_config.root_save_dir)

    def load_weights(self, weights_path):
        self.inference.load_weights(weights_path)

    def test(self, val_path, epoch=0):
        dataloader = get_one_class_val_dataloader(val_path, self.test_task_config)
        self.total_batch_image = len(dataloader)
        self.evaluation.reset()
        self.start_test()
        for index, (images, labels) in enumerate(dataloader):
            prediction, output_list = self.inference.infer(images)
            loss_value = self.compute_loss(output_list, labels)
            self.evaluation.eval(prediction, labels.detach().numpy())
            self.metirc_loss(index, loss_value)
        roc_auc = self.evaluation.get_score()
        self.save_test_value(epoch, roc_auc)
        print("Val epoch loss: {}".format(self.epoch_loss_average.avg))
        return roc_auc, self.epoch_loss_average.avg

    def compute_loss(self, output_list, targets):
        loss = 0
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        targets = targets.to(self.device)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.g_loss_list[0](output_list[0], targets)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.g_loss_list[0](output_list, targets)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.g_loss_list[k](output_list[k], targets)
            else:
                print("compute loss error")
        return loss.item()

    def save_test_value(self, epoch, roc_auc):
        # write epoch results
        with open(self.test_task_config.evaluation_result_path, 'a') as file:
            file.write("Epoch: {} | AUC: {:.3f} \n".format(epoch, roc_auc))
