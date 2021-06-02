#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.data_loader.common.task_dataset_process import TaskDataSetProcess


class SuperResolutionDatasetProcess(TaskDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def normalize_dataset(self, src_lr_image, src_hr_image):
        lr_image = self.normalize_image(src_lr_image)
        hr_image = self.normalize_image(src_hr_image)
        return lr_image, hr_image

    def resize_dataset(self, lr_image, src_image_size, hr_image, target_size):
        image = self.dataset_process.resize(lr_image, src_image_size,
                                            self.resize_type,
                                            pad_color=self.pad_color)
        target = self.dataset_process.resize(hr_image, target_size,
                                             self.resize_type,
                                             pad_color=self.pad_color)
        return image, target
