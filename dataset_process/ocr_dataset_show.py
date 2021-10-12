#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.common.rec_text_process import RecTextProcess
from easyai.data_loader.rec_text.rec_text_sample import RecTextSample
from easyai.helper.image_process import ImageProcess


class OCRDataSetShow():

    def __init__(self, data_path, char_path):
        self.text_process = RecTextProcess()
        character = self.text_process.read_character(char_path)
        self.text_sample = RecTextSample(data_path, "")
        self.text_sample.read_sample(character, 100)
        self.image_process = ImageProcess()

    def show(self):
        for index in range(self.text_sample.get_sample_count()):
            image_path, label = self.text_sample.get_sample_path(index)
            if self.image_process.isImageFile(image_path):
                image = self.image_process.opencvImageRead(image_path)
                txt = label.get_text()

