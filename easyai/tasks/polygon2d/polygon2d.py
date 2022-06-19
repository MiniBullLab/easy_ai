#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.polygon2d.polygon2d_result_process import Polygon2dResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Polygon2d_Task)
class Polygon2d(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Polygon2d_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             class_number=len(self.task_config.detect2d_class))
        self.set_model(gpu_id=gpu_id)
        self.result_process = Polygon2dResultProcess(self.task_config.data['image_size'],
                                                     self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, batch_data in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.timer.tic()
            objects_result = self.single_image_process(batch_data)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))
            if is_show:
                if not self.result_show.show(batch_data['src_image'], objects_result):
                    break
            else:
                pass

    def single_image_process(self, input_data):
        self.set_src_size(input_data['src_image'])
        prediction, _ = self.infer(input_data)
        result = self.result_process.post_process(prediction, self.src_size)
        return result

    def infer(self, batch_data, net_type=0):
        with torch.no_grad():
            input_datas = self.input_datas_processing(batch_data)
            output_list = self.model(input_datas)
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        output = self.common_output(output_list)
        if isinstance(output, (list, tuple)):
            prediction = torch.cat(output, 1)
        else:
            prediction = output
        if prediction is not None:
            prediction = prediction.squeeze(0)
            prediction = prediction.data.cpu().numpy()
        return prediction
