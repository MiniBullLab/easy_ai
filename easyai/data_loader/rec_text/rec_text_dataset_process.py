#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess
from easyai.data_loader.common.rec_text_process import RecTextProcess
from easyai.utility.logger import EasyLogger


class RecTextDataSetProcess(Polygon2dDataSetProcess):

    def __init__(self, char_path, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.text_process = RecTextProcess()
        EasyLogger.debug(char_path)
        self.character = self.text_process.read_character(char_path)

    def normalize_image(self, src_image):
        image = self.dataset_process.normalize(input_data=src_image,
                                               normalize_type=self.normalize_type,
                                               mean=self.mean,
                                               std=self.std)
        image = self.dataset_process.numpy_transpose(image)
        return image

    def normalize_label(self, ocr_object):
        text = ocr_object.get_text()
        text_code, text = self.text_process.text_encode(text)
        result = {'text': text,
                  'targets': text_code}
        return result



