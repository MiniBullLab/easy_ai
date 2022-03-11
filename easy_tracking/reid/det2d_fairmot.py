#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easy_tracking.fairmot.fairmot_post_process import FairMOTPostProcess
from easy_tracking.utility.reid_inference import ReidInference
from easy_tracking.utility.tracking_registry import REGISTERED_REID
from easyai.name_manager.task_name import TaskName


@REGISTERED_REID.register_module(TaskName.Det2D_REID_TASK)
class Det2dFairMOT(ReidInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Det2D_REID_TASK)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             class_number=len(self.task_config.detect2d_class))
        self.set_model(gpu_id=gpu_id)
        self.post_process = FairMOTPostProcess(self.task_config.data['image_size'],
                                               self.task_config.detect2d_class)

    def process(self, src_image, data_type=1, is_show=False):
        input_data = self.get_single_image_data(src_image)
        result, _ = self.single_image_process(input_data)
        return result

    def single_image_process(self, input_data):
        if input_data.get('src_size', None) is not None:
            self.src_size = input_data['src_size'][0].numpy()
        elif input_data.get('src_image', None) is not None:
            self.set_src_size(input_data['src_image'])
        prediction, model_output = self.infer(input_data)
        result = self.post_process(prediction, self.src_size)
        return result, model_output

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
            output = self.common_output(output_list)
        return output, output_list

    def save_result(self, file_path, detection_objects, flag=0):
        path, filename_post = os.path.split(file_path)
        temp_path1, image_dir = os.path.split(path)
        temp_path2, video_dir = os.path.split(temp_path1)
        save_str = os.path.join(video_dir, image_dir, filename_post)
        if flag == 0:
            save_data = save_str + "|"
            for temp_object in detection_objects:
                confidence = temp_object.classConfidence
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
                confidence = temp_object.classConfidence
                x1 = temp_object.min_corner.x
                y1 = temp_object.min_corner.y
                x2 = temp_object.max_corner.x
                y2 = temp_object.max_corner.y
                temp_save_path = os.path.join(self.task_config.save_result_dir, "%s.txt" % temp_object.name)
                with open(temp_save_path, 'a') as file:
                    file.write("{} {} {} {} {} {}\n".format(save_str, confidence, x1, y1, x2, y2))



