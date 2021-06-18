#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
import numpy as np
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.seg.segment_result_process import SegmentResultProcess
from easyai.helper.image_process import ImageProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Segment_Task)
class Segmentation(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Segment_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             points_count=len(self.task_config.segment_class))
        self.set_model(gpu_id=gpu_id)
        self.result_process = SegmentResultProcess(self.task_config.data['image_size'],
                                                   self.task_config.data['resize_type'],
                                                   self.task_config.post_process)
        self.image_process = ImageProcess()

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        os.makedirs(self.task_config.save_result_path, exist_ok=True)

        dataloader = self.get_image_data_lodaer(input_path)
        for index, (file_path, src_image, image) in enumerate(dataloader):
            self.timer.tic()
            self.set_src_size(src_image)
            prediction, _ = self.infer(image)
            _, seg_image = self.result_process.post_process(prediction,
                                                            self.src_size)
            EasyLogger.info('Batch %d Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                if not self.result_show.show(src_image, seg_image,
                                             self.task_config.segment_class):
                    break
            else:
                self.save_result_confidence(file_path, prediction)
                self.save_result(file_path, seg_image)

    def save_result(self, file_path, seg_image):
        path, filename_post = os.path.split(file_path)
        filename, post = os.path.splitext(filename_post)
        save_result_path = os.path.join(self.task_config.save_result_path,
                                        "%s.png" % filename)
        self.image_process.opencv_save_image(save_result_path, seg_image)

    def save_result_confidence(self, file_path, prediction):
        if prediction.ndim == 2:
            path, filename_post = os.path.split(file_path)
            filename, post = os.path.splitext(filename_post)
            save_result_path = os.path.join(self.task_config.save_result_path, "%s.txt" % filename)
            np.savetxt(save_result_path, prediction, fmt='%0.8f')

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list[:])
        return output, output_list

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = np.squeeze(prediction.data.cpu().numpy())
        return prediction

