#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import torch
from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.rec_text.text_result_process import TextResultProcess
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_INFERENCE_TASK.register_module(TaskName.RecognizeText)
class RecognizeText(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.RecognizeText)
        self.set_model_param(data_channel=self.task_config.data['data_channel'],
                             class_number=self.task_config.character_count)
        self.set_model(gpu_id=gpu_id)
        self.result_process = TextResultProcess(self.task_config.character_set,
                                                self.task_config.post_process)

    def process(self, input_path, data_type=1, is_show=False):
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, batch_data in enumerate(dataloader):
            self.timer.tic()
            self.set_src_size(batch_data['src_image'])
            text_objects = self.single_image_process(batch_data)
            EasyLogger.debug('Batch %d/%d Done. (%.3fs)' % (i, image_count,
                                                            self.timer.toc()))
            if is_show:
                print(batch_data['file_path'], text_objects[0].get_text())
            else:
                self.save_result(batch_data['file_path'], text_objects)

    def save_result(self, file_path, ocr_object):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{} {} \n".format(filename_post, ocr_object[0].get_text()))

    def single_image_process(self, input_data):
        prediction, _ = self.infer(input_data)
        result = self.result_process.post_process(prediction)
        return result

    def infer(self, input_data, net_type=0):
        with torch.no_grad():
            image_data = input_data['image'].to(self.device)
            output_list = self.model(image_data)
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
            EasyLogger.error("compute loss error")
        if prediction is not None:
            prediction = prediction.squeeze(0)
            prediction = prediction.data.cpu().numpy()
        return prediction
