#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Detect2d_Task)
class Detection2d(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Detect2d_Task)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             class_number=len(self.task_config.detect2d_class))
        self.set_model(gpu_id=gpu_id)
        self.result_process = Detect2dResultProcess(self.task_config.data['image_size'],
                                                    self.task_config.detect2d_class,
                                                    self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path)
        for i, batch_data in enumerate(dataloader):
            self.timer.tic()
            result, _ = self.single_image_process(batch_data)
            EasyLogger.info('Batch %d Done. (%.3fs)' % (i, self.timer.toc()))
            if is_show:
                if not self.result_show.show(batch_data['src_image'], result):
                    break
            else:
                self.save_result(batch_data['file_path'], result, 0)

    def single_image_process(self, input_data):
        if input_data.get('src_size', None) is not None:
            self.src_size = input_data['src_size'][0].numpy()
        elif input_data.get('src_image', None) is not None:
            self.set_src_size(input_data['src_image'])
        prediction, model_output = self.infer(input_data)
        detection_objects = self.result_process.post_process(prediction,
                                                             self.src_size)
        return detection_objects, model_output

    def save_result(self, file_path, detection_objects, flag=0):
        path, filename_post = os.path.split(file_path)
        if flag == 0:
            save_data = filename_post + "|"
            for temp_object in detection_objects:
                confidence = temp_object.classConfidence * temp_object.objectConfidence
                x1 = temp_object.min_corner.x
                y1 = temp_object.min_corner.y
                x2 = temp_object.max_corner.x
                y2 = temp_object.max_corner.y
                save_data = save_data + "{} {} {} {} {} {}|".format(temp_object.name,
                                                                    confidence,
                                                                    x1, y1, x2, y2)
            save_data += "\n"
            with open(self.task_config.save_result_path, 'a') as file:
                file.write(save_data)
        elif flag == 1:
            if self.task_config.save_result_dir is not None and \
                    not os.path.exists(self.task_config.save_result_dir):
                os.makedirs(self.task_config.save_result_dir, exist_ok=True)
            for temp_object in detection_objects:
                confidence = temp_object.classConfidence * temp_object.objectConfidence
                x1 = temp_object.min_corner.x
                y1 = temp_object.min_corner.y
                x2 = temp_object.max_corner.x
                y2 = temp_object.max_corner.y
                temp_save_path = os.path.join(self.task_config.save_result_dir, "%s.txt" % temp_object.name)
                with open(temp_save_path, 'a') as file:
                    file.write("{} {} {} {} {} {}\n".format(filename_post, confidence, x1, y1, x2, y2))

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
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
        return prediction



