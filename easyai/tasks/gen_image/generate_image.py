#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.gen_image.generate_image_result_process import GenerateImageResultProcess
from easyai.helper.imageProcess import ImageProcess
from easyai.visualization.task_show.image_show import ImageShow
from easyai.base_name.task_name import TaskName


class GenerateImage(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.GenerateImage)

        self.model_args['image_size'] = len(self.task_config.image_size)
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        self.device = self.torchModelProcess.get_device()

        self.result_process = GenerateImageResultProcess(self.task_config.image_size)
        self.result_show = ImageShow()
        self.image_process = ImageProcess()

    def process(self, input_path, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        os.makedirs(self.task_config.save_result_path, exist_ok=True)
        dataloader = self.get_image_data_lodaer(input_path)
        for index, (file_path, src_image, image) in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(image)
            result = self.postprocess(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                if not self.result_show.show(result):
                    break
            else:
                self.save_result(file_path, result)

    def save_result(self, file_path, prediction):
        path, filename_post = os.path.split(file_path)
        filename, post = os.path.splitext(filename_post)
        save_result_path = os.path.join(self.task_config.save_result_path, "%s.png" % filename)
        self.image_process.opencv_save_image(save_result_path, prediction)

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            fake_images = self.model.generator_input_data(input_data, 1)
            output_list = self.model(fake_images.to(self.device))
            output = self.compute_output(output_list)
        return output, output_list

    def postprocess(self, result):
        result_image = None
        if result is not None:
            result_image = self.result_process.get_result_image(result,
                                                                self.task_config.post_prcoess_type)
        return result_image

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
