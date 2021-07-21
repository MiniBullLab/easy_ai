#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_inference import BaseInference
from easyai.tasks.polygon2d.polygon2d import Polygon2d
from easyai.tasks.rec_text.recognize_text import RecognizeText
from easyai.helper.data_structure import OCRObject
from easyai.data_loader.common.ocr_dataloader import OCRLoader
from easyai.name_manager.task_name import TaskName
from easyai.tasks.utility.task_registry import REGISTERED_INFERENCE_TASK
from easyai.utility.logger import EasyLogger


@REGISTERED_INFERENCE_TASK.register_module(TaskName.OCR_Task)
class OCRTask(BaseInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.OCR_Task)
        self.det2d_inference = Polygon2d(model_name[0], gpu_id, self.task_config.det_config)
        self.text_inference = RecognizeText(model_name[1], gpu_id, self.task_config.text_config)

    def load_weights(self, weights_path):
        self.det2d_inference.load_weights(weights_path[0])
        self.text_inference.load_weights(weights_path[1])

    def process(self, input_path, data_type=1, is_show=False):
        self.task_config = self.det2d_inference.task_config
        dataloader = self.get_image_data_lodaer(input_path)
        image_count = len(dataloader)
        for i, batch_data in enumerate(dataloader):
            print('%g/%g' % (i + 1, image_count), end=' ')
            self.timer.tic()
            self.set_src_size(batch_data['src_image'])
            detection_objects = self.det2d_inference.single_image_process(self.src_size,
                                                                          batch_data)
            ocr_result = self.convert_ocr_object(detection_objects)

            ocr_dataloader = OCRLoader(ocr_result, batch_data['src_image'],
                                       self.text_inference.task_config.data['image_size'],
                                       self.text_inference.task_config.data['data_channel'],
                                       self.text_inference.task_config.data['resize_type'],
                                       self.text_inference.task_config.data['normalize_type'],
                                       self.text_inference.task_config.data['mean'],
                                       self.text_inference.task_config.data['std'])
            for index, ocr_data in enumerate(ocr_dataloader):
                text = self.text_inference.single_image_process(ocr_data)
                ocr_result[index].set_text(text[0].get_text())
                ocr_result[index].text_confidence = text[0].text_confidence
            if not self.result_show.show(batch_data['src_image'], ocr_result):
                break

    def infer(self, input_data, net_type=0):
        pass

    def convert_ocr_object(self, detection_objects):
        result = []
        for temp_object in detection_objects:
            temp = OCRObject()
            temp.copy_polygon(temp_object)
            result.append(temp)
        return result

