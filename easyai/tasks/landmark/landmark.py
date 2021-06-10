#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.landmark.landmark_result_process import LandmarkResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Landmark)
class Landmark(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Landmark)
        self.set_model_param(data_channel=self.task_config.data_channel,
                             points_count=self.task_config.points_count)
        self.set_model(gpu_id=gpu_id)
        self.result_process = LandmarkResultProcess(self.task_config.points_count,
                                                    self.task_config.image_size,
                                                    self.task_config.post_prcoess)

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.timer.tic()
            self.set_src_size(src_image)
            objects_pose = self.single_image_process(self.src_size, img)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))
            if is_show:
                if not self.result_show.show(src_image, [objects_pose], self.task_config.skeleton):
                    break
            else:
                pass

    def single_image_process(self, src_size, input_image):
        prediction, _ = self.infer(input_image)
        _, pose = self.result_process.post_process(prediction, src_size)
        return pose

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        count = len(output_list)
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        prediction = None
        if loss_count == 1 and output_count == 1:
            prediction = self.model.lossList[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            prediction = self.model.lossList[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            preds = []
            for i in range(0, count):
                temp = self.model.lossList[i](output_list[i])
                preds.append(temp)
            prediction = torch.cat(preds, 1)
        else:
            print("compute loss error")
        if isinstance(prediction, (tuple, list)):
            result = []
            for pre in prediction:
                pre = pre.squeeze(0)
                pre = pre.data.cpu().numpy()
                result.append(pre)
            prediction = tuple(result)
        elif prediction is not None:
            prediction = prediction.squeeze(0)
            prediction = prediction.data.cpu().numpy()
        return prediction
