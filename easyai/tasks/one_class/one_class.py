#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.one_class.one_class_result_process import OneClassResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.OneClass)
class OneClass(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.OneClass)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             image_size=self.task_config.data['image_size'])
        self.set_model(gpu_id=gpu_id)
        self.result_process = OneClassResultProcess(self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path)
        for index, batch_data in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(batch_data)
            class_index, class_confidence = self.result_process.post_process(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                if not self.result_show.show(batch_data['src_image'],
                                             class_index,
                                             self.task_config.class_name):
                    break
            else:
                self.save_result(batch_data['file_path'], class_index, class_confidence)

    def save_result(self, file_path, class_index, class_confidence):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {} {:.5f}\n".format(filename_post,
                                               class_index,
                                               class_confidence))

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            model_output = self.model(image_data)
            output = self.compute_output(model_output)
        return output, model_output

    def compute_output(self, model_output):
        output = None
        prediction = None
        if self.task_config.model_type == 0:
            output = self.common_output(model_output)
        elif self.task_config.model_type == 1:
            output = self.gan_output(model_output)
        if isinstance(output, (list, tuple)):
            prediction = [np.squeeze(x.data.cpu().numpy()) for x in output]
        elif output is not None:
            prediction = np.squeeze(output.data.cpu().numpy())
        return prediction

