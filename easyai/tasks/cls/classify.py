#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.cls.classify_result_process import ClassifyResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Classify_Task)
class Classify(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Classify_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             class_number=len(self.task_config.class_name))
        self.set_model(gpu_id=gpu_id)
        self.result_process = ClassifyResultProcess(self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path)
        for index, batch_data in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(batch_data)
            class_index, class_confidence = self.result_process.post_process(prediction)
            EasyLogger.info('Batch %d Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                if not self.result_show.show(batch_data['src_image'],
                                             class_index[0].cpu().numpy(),
                                             self.task_config.class_name):
                    break
            else:
                output_count = prediction.size(1)
                if output_count == 1:
                    batch_size = prediction.size(0)
                    class_index = torch.ones(batch_size)
                self.save_result(batch_data['file_path'], class_index,
                                 class_confidence)

    def save_result(self, file_path, class_index, class_confidence):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {} {:.5f}\n".format(filename_post,
                                               class_index[0].cpu().numpy(),
                                               class_confidence[0][0].cpu().numpy()))

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        output = self.common_output(output_list)
        return output

