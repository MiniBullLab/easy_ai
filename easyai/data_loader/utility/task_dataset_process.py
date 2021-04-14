#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class TaskDataSetProcess(BaseDataSetProcess):

    def __init__(self, resize_type, normalize_type, mean, std, pad_color):
        super().__init__()
        self.resize_type = resize_type
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pad_color = pad_color
        self.dataset_process = ImageDataSetProcess()

    def normalize_image(self, src_image):
        if self.normalize_type < 0:
            torchvision_transform = self.torchvision_process.torch_normalize(normalize_type=self.normalize_type,
                                                                             mean=self.mean,
                                                                             std=self.std)
            image = torchvision_transform(src_image)
        else:
            image = self.dataset_process.normalize(input_data=src_image,
                                                   normalize_type=self.normalize_type,
                                                   mean=self.mean,
                                                   std=self.std)
            image = self.dataset_process.numpy_transpose(image)
            image = self.numpy_to_torch(image, flag=0)
        return image

    def resize_image(self, src_image, image_size):
        image = self.dataset_process.resize(src_image, image_size,
                                            self.resize_type,
                                            pad_color=self.pad_color)
        return image

    def get_roi_image(self, src_image, expand_box):
        xmin = expand_box.min_corner.x
        ymin = expand_box.min_corner.y
        xmax = expand_box.max_corner.x
        ymax = expand_box.max_corner.y
        if len(src_image.shape) == 3:
            image = src_image[ymin:ymax, xmin:xmax, :]
        elif len(src_image.shape) == 2:
            image = src_image[ymin:ymax, xmin:xmax]
        else:
            image = None
        return image
