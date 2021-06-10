#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.gen_image.generate_image_result_process import GenerateImageResultProcess
from easyai.helper.image_process import ImageProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.GenerateImage)
class GenerateImage(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.GenerateImage)
        self.set_model_param(data_channel=self.task_config.data_channel,
                             image_size=self.task_config.image_size)
        self.set_model(gpu_id=gpu_id)
        self.result_process = GenerateImageResultProcess(self.task_config.post_prcoess,
                                                         self.task_config.image_size)
        self.image_process = ImageProcess()
        self.save_index = 0

    def process(self, input_path, data_type=1, is_show=False):
        if data_type == 0:
            batch_data = torch.randn((1, 1))
            self.timer.tic()
            prediction, _ = self.infer(batch_data)
            result = self.result_process.post_process(prediction)
            print('Done. (%.3fs)' % (self.timer.toc()))
            if is_show:
                if not self.result_show.show(result):
                    pass
            else:
                file_path = "%d_generate_image.png" % self.save_index
                self.save_index += 1
                self.save_result(file_path, result)
        else:
            os.system('rm -rf ' + self.task_config.save_result_path)
            dataloader = self.get_image_data_lodaer(input_path)
            for index, (file_path, src_image, image) in enumerate(dataloader):
                self.timer.tic()
                prediction, _ = self.infer(image)
                result = self.result_process.post_process(prediction)
                print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
                if is_show:
                    if not self.result_show.show(result):
                        break
                else:
                    self.save_result(file_path, result)

    def save_result(self, file_path, prediction):
        os.makedirs(self.task_config.save_result_path, exist_ok=True)
        path, filename_post = os.path.split(file_path)
        filename, post = os.path.splitext(filename_post)
        save_result_path = os.path.join(self.task_config.save_result_path, "%s.png" % filename)
        self.image_process.opencv_save_image(save_result_path, prediction)

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            fake_images = self.model.generator_input_data(input_data)
            output_list = self.model(fake_images.to(self.device))
            output = self.compute_output(output_list)
        return output, output_list

    def compute_output(self, output_list):
        output = None
        prediction = None
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            output = self.model.g_loss_list[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            output = self.model.g_loss_list[0](output_list)
        if output is not None:
            prediction = np.squeeze(output.data.cpu().numpy())
        return prediction
