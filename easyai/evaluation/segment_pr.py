#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import os
import numpy as np
from sklearn.metrics import confusion_matrix
from easyai.helper.image_process import ImageProcess
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.data_loader.seg.segment_sample import SegmentSample


class SegmentPrecisionAndRecall():

    def __init__(self, seg_label_type=0, segment_class=None):
        self.seg_label_type = seg_label_type
        self.segment_class = segment_class
        self.image_process = ImageProcess()
        self.label_converter = ConvertSegmentionLable()
        self.segment_sample = SegmentSample(None)
        self.confidence_file_post = ".txt"

    def eval(self, result_dir, val_path):
        gt_data_list = self.segment_sample.get_image_and_label_list(val_path)
        for image_path, label_path in gt_data_list:
            path, filename_post = os.path.split(label_path)
            filename, post = os.path.splitext(filename_post)
            test_filename = filename + self.confidence_file_post
            test_path = os.path.join(result_dir, test_filename)

    def calculate_roc_value(self, result_dir, gt_data_list):
        result = []
        for threshold in range(0, 100, 1):
            threshold = threshold / 100.0
            tp = 0  # 前景像素点中被正确标记为前景像素的数目
            fp = 0  # 背景像素点中被错误标记为前景像素的数目
            fn = 0  # 前景像素点中被错误标记为背景像素的数目
            tn = 0  # 背景像素点中被正确标记为背景像素的数目
            for image_path, label_path in gt_data_list:
                path, filename_post = os.path.split(label_path)
                filename, post = os.path.splitext(filename_post)
                test_filename = filename + self.confidence_file_post
                test_path = os.path.join(result_dir, test_filename)
                prediction = self.read_test_data(test_path)
                result = (prediction >= threshold).astype(int)
                gt_mask = self.read_label_image(label_path, self.seg_label_type)
                cm = confusion_matrix(gt_mask.reshape(-1), result.reshape(-1), labels=[0, 1])
                # print(cm)
                # print(cm.ravel())
                tn += cm[0][0]
                fn += cm[1][0]
                tp += cm[1][1]
                fp += cm[0][1]

            recall = tp * 1.0 / (tp + fn + 1e-8)
            precision = tp * 1.0 / (tp + fp + 1e-8)
            result.append((precision, recall))
        return result

    def read_label_image(self, label_path, label_type):
        if label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.label_converter.process_segment_label(label_path,
                                                              label_type,
                                                              self.segment_class)
        return mask

    def read_test_data(self, test_path):
        result = np.loadtxt(test_path, dtype=np.float32)
        return result

