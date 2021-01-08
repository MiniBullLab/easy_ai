#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.visualization.task_show.classify_show import ClassifyShow
from easyai.base_name.task_name import TaskName
from easyai.tasks.utility.registry import REGISTERED_INFERENCE_TASK


@REGISTERED_INFERENCE_TASK.register_module(TaskName.Classify_Task)
class Classify(BaseInference):

    def __init__(self, cfg_path, gpu_id, config_path=None):
        super().__init__(cfg_path, config_path, TaskName.Classify_Task)
        self.model_args['class_number'] = len(self.task_config.class_name)
        self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        self.result_show = ClassifyShow()

    def process(self, input_path, data_type=1, is_show=False):
        os.system('rm -rf ' + self.task_config.save_result_path)
        dataloader = self.get_image_data_lodaer(input_path)
        for index, (file_path, src_image, image) in enumerate(dataloader):
            self.timer.tic()
            prediction, _ = self.infer(image)
            class_index, class_confidence = self.postprocess(prediction)
            print('Batch %d... Done. (%.3fs)' % (index, self.timer.toc()))
            if is_show:
                if not self.result_show.show(src_image,
                                             class_index[0].cpu().numpy(),
                                             self.task_config.class_name):
                    break
            else:
                self.save_result(file_path, class_index, class_confidence)

    def save_result(self, file_path, class_index, class_confidence):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {} {}\n".format(filename_post,
                                           class_index[0].cpu().numpy(),
                                           class_confidence[0].cpu().numpy()))

    def infer(self, input_data, threshold=0.0):
        with torch.no_grad():
            output_list = self.model(input_data.to(self.device))
            output = self.compute_output(output_list)
        return output, output_list

    def postprocess(self, result):
        class_indices = torch.argmax(result, dim=1)
        class_confidence = result[class_indices]
        return class_indices, class_confidence

    def compute_output(self, output_list):
        output = None
        if len(output_list) == 1:
            output = self.model.lossList[0](output_list[0])
        return output

