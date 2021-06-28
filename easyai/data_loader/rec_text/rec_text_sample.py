#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample


class RecTextSample(BaseDetectionSample):

    def __init__(self, train_path, language):
        super().__init__()
        self.train_path = train_path
        self.language = language
        self.max_length = 0
        self.image_and_ocr_list = []
        self.sample_count = 0

    def read_sample(self, char_list):
        image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.image_and_ocr_list = self.get_image_and_ocr_list(image_and_label_list,
                                                              char_list)
        filtering_result = []
        for image_path, ocr_object in self.image_and_ocr_list:
            for char in ocr_object.object_text:
                if char in char_list:
                    filtering_result.append((image_path, ocr_object))
                    break
        self.image_and_ocr_list = filtering_result

        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, ocr_object = self.image_and_ocr_list[temp_index]
        return img_path, ocr_object

    def get_sample_count(self):
        return len(self.image_and_ocr_list)

    def get_image_and_ocr_list(self, image_and_label_list, char_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, ocr_objects = self.json_process.parse_ocr_data(label_path)
            for ocr in ocr_objects:
                if ocr.language.strip() in self.language:
                    self.max_length = max(len(ocr.object_text), self.max_length)
                    result.append((image_path, ocr))
        return result
