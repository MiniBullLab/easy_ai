#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.det2d.detect2d_result_process import Detect2dResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Detect2d_Task)
class Detection2d(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Detect2d_Task)
        self.set_model_param(data_channel=self.task_config.data_channel,
                             class_number=len(self.task_config.detect2d_class))
        self.set_model(gpu_id=gpu_id)
        self.result_process = Detect2dResultProcess(self.task_config.post_prcoess_type,
                                                    self.task_config.nms_th,
                                                    self.task_config.image_size,
                                                    self.task_config.detect2d_class)

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path)
        for i, (file_path, src_image, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            self.timer.tic()
            self.set_src_size(src_image)
            result = self.single_image_process(self.src_size, img)
            print('Batch %d... Done. (%.3fs)' % (i, self.timer.toc()))
            if is_show:
                if not self.result_show.show(src_image, result):
                    break
            else:
                self.save_result(file_path, result, 0)

    def single_image_process(self, src_size, input_image):
        prediction, _ = self.infer(input_image)
        detection_objects = self.result_process.postprocess(prediction,
                                                            src_size,
                                                            self.task_config.confidence_th)
        return detection_objects

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
        if prediction is not None:
            prediction = prediction.squeeze(0)
        return prediction



