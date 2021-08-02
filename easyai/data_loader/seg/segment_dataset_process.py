#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import numpy as np
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class SegmentDatasetProcess(TaskDataSetProcess):
    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.label_pad_color = 250

    def resize_dataset(self, src_image, image_size, label):
        assert src_image.shape[:2] == label.shape[:2]
        image = self.resize_image(src_image, image_size)
        target = self.resize_lable(label, image_size)
        return image, target

    def resize_lable(self, label, dst_size):
        target = self.dataset_process.resize(label, dst_size, self.resize_type,
                                             pad_color=self.label_pad_color)
        if target is not None:
            target = np.array(target, dtype=np.uint8)
        return target

    def change_label(self, label, number_class):
        valid_masks = np.zeros(label.shape)
        for index in range(0, number_class):
            valid_mask = label == index  # set false to position of seg that not in valid label
            valid_masks += valid_mask  # set 0 to position of seg that not in valid label
        valid_masks[valid_masks == 0] = -1
        mask = np.float32(label) * valid_masks
        mask[mask < 0] = self.label_pad_color
        mask = np.uint8(mask)
        return mask
