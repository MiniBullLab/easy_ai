#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.base_data_loader import *
from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess


class OCRLoader(DataLoader):

    def __init__(self, object_list, src_image, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(None, data_channel)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.dataset_process = Polygon2dDataSetProcess(resize_type, normalize_type,
                                                       mean, std,
                                                       pad_color=self.get_pad_color())
        self.src_image = src_image
        self.object_list = object_list
        self.count = len(self.object_list)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        temp_object = self.object_list[self.index]
        image = self.dataset_process.get_rotate_crop_image(self.src_image,
                                                           temp_object.get_polygon()[:])
        image = self.dataset_process.resize_image(image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        torch_image = torch_image.unsqueeze(0)
        return torch_image

    def __len__(self):
        return self.count
