#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.cls.classify_result_process import ClassifyResultProcess
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger

from easy_pc.tasks.utility.base_pc_inference import BasePCInference
from easy_pc.name_manager.pc_task_name import PCTaskName


@REGISTERED_INFERENCE_TASK.register_module(PCTaskName.PC_Classify_Task)
class PointCloudClassify(BasePCInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, PCTaskName.PC_Classify_Task)
        self.set_model_param(data_channel=self.task_config.data['point_features'],
                             class_number=len(self.task_config.class_name))
        self.set_model(gpu_id=gpu_id)
        self.result_process = ClassifyResultProcess(self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_point_cloud_data_lodaer(input_path)
        for index, batch_data in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(batch_data)
            class_index, class_confidence = self.result_process.post_process(prediction)
            EasyLogger.info('Batch %d Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                break
            else:
                self.save_result(batch_data['file_path'], class_index,
                                 class_confidence)

    def save_result(self, file_path, class_index, class_confidence):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {} {:.5f}\n".format(filename_post,
                                               class_index,
                                               class_confidence))

    def single_image_process(self, input_data):
        pass

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            pc_data = input_data['point_cloud'].to(self.device)
            output_list = self.model(pc_data)
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        output = None
        if len(output_list) == 1:
            output = self.model.lossList[0](output_list[0])
        return output

