#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample


class Pose2dSample(BaseDetectionSample):

    def __init__(self, data_path, class_name):
        super().__init__()
        self.data_path = data_path
        self.class_name = class_name

        self.image_and_box_list = []
        self.sample_count = 0

    def read_sample(self):
        image_and_label_list = self.get_image_and_label_list(self.data_path)
        self.image_and_box_list = self.get_image_and_box_list(image_and_label_list)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, box = self.image_and_box_list[temp_index]
        return img_path, box

    def get_sample_count(self):
        return len(self.image_and_box_list)

    def get_image_and_box_list(self, image_and_label_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, boxes = self.json_process.parse_pose2d_data(label_path)
            for box in boxes:
                if box.name in self.class_name:
                    result.append((image_path, box))
        return result
