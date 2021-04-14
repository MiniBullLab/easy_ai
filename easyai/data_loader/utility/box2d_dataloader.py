#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.utility.box2d_dataset_process import Box2dDataSetProcess


class Box2dLoader(DataLoader):

    def __init__(self, box_list, src_image, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(data_channel)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.dataset_process = Box2dDataSetProcess(resize_type, normalize_type,
                                                   mean, std,
                                                   pad_color=self.get_pad_color())
        self.src_image = src_image
        self.box_list = box_list
        self.count = len(self.box_list)

        self.expand_ratio = 0.15

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        box = self.box_list[self.index]
        src_size = (self.src_image.shape[1], self.src_image.shape[0])  # [width, height]
        expand_box = self.dataset_process.get_expand_box(src_size, box, self.expand_ratio)
        image = self.dataset_process.get_roi_image(self.src_image, expand_box)
        image = self.dataset_process.resize_image(image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        torch_image = torch_image.unsqueeze(0)
        return box, torch_image

    def __len__(self):
        return self.count