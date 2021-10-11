#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.keypoint2d.keypoint2d_result_process import KeyPoint2dResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.KeyPoint2d_Task)
class KeyPoint2d(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.KeyPoint2d_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'])
        self.set_model(gpu_id=gpu_id)
        self.result_process = KeyPoint2dResultProcess(self.task_config.data['image_size'],
                                                      self.task_config.points_count,
                                                      self.task_config.points_class,
                                                      self.task_config.post_prcoess)

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        for i, batch_data in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(batch_data['src_image'])

            self.timer.tic()
            prediction = self.infer(batch_data)
            _, result_objects = self.result_process.post_process(prediction,
                                                                 self.src_size)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(batch_data['src_image'],
                                         result_objects, self.task_config.skeleton):
                break

    def single_image_process(self, input_data):
        pass

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
            output = self.compute_output(output_list)
        return output

    def compute_output(self, output_list):
        output = self.common_output(output_list)
        if isinstance(output, (list, tuple)):
            prediction = torch.cat(output, 1)
        else:
            prediction = output
        if prediction is not None:
            prediction = prediction.squeeze(0)
        return prediction
