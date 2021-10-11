#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
import numpy as np
from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample
from easyai.utility.logger import EasyLogger


class DetectionSample(BaseDetectionSample):

    def __init__(self, train_path, class_name, is_balance=False):
        super().__init__()
        self.train_path = train_path
        self.is_blance = is_balance
        self.class_name = class_name

        self.is_shuffled = False
        self.shuffled_vectors = {}
        self.balanced_files = {}
        self.balance_file_count = {}
        self.image_and_label_list = []
        self.sample_count = 0
        self.balanced_file_index = np.zeros(len(self.class_name))

    def read_sample(self):
        try:
            if self.is_blance:
                self.balanced_files, self.balance_file_count = \
                    self.get_blance_file_list(self.train_path, self.class_name)
            else:
                self.image_and_label_list = self.get_image_and_label_list(self.train_path)
            self.sample_count = self.get_sample_count()
            EasyLogger.warn("%s sample count: %d" % (self.train_path,
                                                     self.sample_count))
        except ValueError as err:
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(err)

    def get_sample_boxes(self, label_path):
        result = []
        _, boxes = self.json_process.parse_rect_data(label_path)
        for box in boxes:
            if box.name in self.class_name:
                result.append(box)
        return result

    def get_sample_path(self, index, class_index=None):
        if self.is_shuffled:
            if self.is_blance:
                name = self.class_name[class_index]
                files = self.balanced_files[name]
                temp_index = index % self.balance_file_count[name]
                temp_index = self.shuffled_vectors[name][temp_index]
                img_path, label_path = files[temp_index]
            else:
                temp_index = index % self.sample_count
                temp_index = self.shuffled_vectors[temp_index]
                img_path, label_path = self.image_and_label_list[temp_index]
        else:
            if self.is_blance:
                name = self.class_name[class_index]
                files = self.balanced_files[name]
                temp_index = index % self.balance_file_count[name]
                img_path, label_path = files[temp_index]
            else:
                temp_index = index % self.sample_count
                img_path, label_path = self.image_and_label_list[temp_index]
        return img_path, label_path

    def get_sample_start_index(self, index, batch_size, class_index=None):
        if self.is_blance:
            start_index = self.balanced_file_index[class_index]
            name = self.class_name[class_index]
            self.balanced_file_index[class_index] = (start_index + batch_size) % \
                                                    self.balance_file_count[name]
        else:
            start_index = index * batch_size
        return int(start_index)

    def get_sample_count(self):
        result = 0
        if self.is_blance:
            for key, value in self.balance_file_count.items():
                result += value
        else:
            result = len(self.image_and_label_list)
        return result

    def shuffle_sample(self):
        self.shuffled_vectors = {}
        if self.is_blance:
            self.balanced_file_index = np.zeros(len(self.class_name))
            for i in range(0, len(self.class_name)):
                self.shuffled_vectors[self.class_name[i]] = \
                    np.random.permutation(self.balance_file_count[self.class_name[i]])
        else:
            self.shuffled_vectors = np.random.permutation(self.sample_count)
        self.is_shuffled = True

    def get_blance_file_list(self, train_path, class_name):
        file_list = {}
        file_count = {}
        class_count = len(class_name)
        path, _ = os.path.split(train_path)
        for i in range(0, class_count):
            class_file = self.class_name[i] + ".txt"
            class_path = os.path.join(path, class_file)
            file_list[class_name[i]] = self.get_image_and_label_list(class_path)
            file_count[class_name[i]] = len(file_list[class_name[i]])
        return file_list, file_count
