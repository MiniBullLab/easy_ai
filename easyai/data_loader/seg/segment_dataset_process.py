#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import numpy as np
from easyai.data_loader.common.task_dataset_process import TaskDataSetProcess


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
        target = None
        if self.resize_type == 0:
            target = self.dataset_process.cv_image_resize(label, dst_size)
            target = np.array(target, dtype=np.uint8)
        elif self.resize_type == 1:
            src_size = (label.shape[1], label.shape[0])  # [width, height]
            ratio, pad_size = self.dataset_process.get_square_size(src_size, dst_size)
            target = self.dataset_process.image_resize_square(label, ratio, pad_size,
                                                              self.label_pad_color)
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
