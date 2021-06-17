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
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.set_src_size(src_image)

            self.timer.tic()
            prediction = self.infer(img)
            _, result_objects = self.result_process.post_process(prediction,
                                                                 self.src_size)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))

            if not self.result_show.show(src_image, result_objects, self.task_config.skeleton):
                break

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction
