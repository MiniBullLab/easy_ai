#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.helper.dir_process import DirProcess
from easyai.data_loader.cls.classify_sample import ClassifySample


class ClassifyPrecisionAndRecall():

    def __init__(self, class_number):
        self.class_number = class_number
        self.dir_process = DirProcess()
        self.classify_sample = ClassifySample(None)

    def eval(self, result_path, val_path):
        result = {}
        result_data_list = self.get_result(result_path)
        gt_data_list = self.classify_sample.get_image_and_label_list(val_path)
        for class_index in range(self.class_number):
            class_result_data_list = [x for x in result_data_list if x[1] == class_index]
            class_gt_data_list = [x for x in gt_data_list if x[1] == class_index]
            pr_value = self.calculate_roc_value(class_result_data_list, class_gt_data_list)
            result[class_index] = pr_value
        return result

    def calculate_roc_value(self, result_data_list, gt_data_list):
        result = []
        for threshold in range(0, 100, 1):
            threshold = threshold / 100.0
            input_data_list = [x for x in result_data_list if x[2] >= threshold]
            pecision, recall = self.calculate_pr(input_data_list, gt_data_list)
            result.append((pecision, recall))
        return result

    def calculate_pr(self, result_data_list, gt_data_list):
        TP = 0
        FN = 0
        pecision = 0
        recall = 0
        result_count = len(result_data_list)
        gt_count = len(gt_data_list)
        for data in result_data_list:
            if self.has_in_images(data[0], gt_data_list):
                TP += 1

        for data in gt_data_list:
            path, image_name = os.path.split(data[0])
            if not self.has_in_images(image_name.strip(), result_data_list):
                FN += 1
        if result_count != 0:
            pecision = float(TP) / result_count
        if gt_count != 0:
            recall = 1 - float(FN) / gt_count
        return pecision, recall

    def has_in_images(self, image_name, data_list):
        result = False
        for image_data in data_list:
            if image_name in image_data[0]:
                result = True
                break
        return result

    def get_result(self, result_path):
        result = []
        if not os.path.exists(result_path):
            return result
        for line_data in self.dir_process.getFileData(result_path):
            split_datas = [x.strip() for x in line_data.split(' ') if x.strip()]
            filename_post = split_datas[0].strip()
            class_index = int(split_datas[1])
            class_confidence = float(split_datas[2])
            # print(filename_post)
            result.append((filename_post, class_index, class_confidence))
        return result
