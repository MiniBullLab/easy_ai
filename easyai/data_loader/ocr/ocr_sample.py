#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample


class OCRSample(BaseDetectionSample):

    def __init__(self, train_path, language=""):
        super().__init__()
        self.train_path = train_path
        self.language = language

        self.image_and_ocr_list = []
        self.sample_count = 0

    def read_sample(self):
        image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.image_and_ocr_list = self.get_image_and_ocr_list(image_and_label_list)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, ocr_objects = self.image_and_ocr_list[temp_index]
        return img_path, ocr_objects

    def get_sample_count(self):
        return len(self.image_and_ocr_list)

    def get_image_and_ocr_list(self, image_and_label_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, ocr_objects = self.json_process.parse_ocr_data(label_path)
            filter_objects = []
            for ocr in ocr_objects:
                if not self.language:
                    filter_objects.append(ocr)
                elif ocr.language.strip() in self.language:
                    filter_objects.append(ocr)
            result.append((image_path, filter_objects))
        return result
