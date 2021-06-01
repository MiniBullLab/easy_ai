#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.base_classify_sample import BaseClassifySample


class GenImageSample(BaseClassifySample):

    def __init__(self, train_path):
        super().__init__()
        self.train_path = train_path
        self.data_and_label_list = []
        self.sample_count = 0

    def read_sample(self):
        self.data_and_label_list = self.get_image_and_label_list(self.train_path)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, label = self.data_and_label_list[temp_index]
        return img_path, label

    def get_sample_count(self):
        return len(self.data_and_label_list)

