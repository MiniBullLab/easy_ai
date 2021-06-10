#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample


class RecTextSample(BaseDetectionSample):

    def __init__(self, train_path, language):
        super().__init__()
        self.train_path = train_path
        self.language = language

        self.image_and_polygon_list = []
        self.sample_count = 0

    def read_sample(self):
        image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.image_and_polygon_list = self.get_image_and_polygon_list(image_and_label_list)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, polygon = self.image_and_polygon_list[temp_index]
        return img_path, polygon

    def get_sample_count(self):
        return len(self.image_and_polygon_list)

    def get_image_and_polygon_list(self, image_and_label_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, ocr_objects = self.json_process.parse_ocr_data(label_path)
            for ocr in ocr_objects:
                if ocr.language.strip() in self.language:
                    result.append((image_path, ocr))
        return result
