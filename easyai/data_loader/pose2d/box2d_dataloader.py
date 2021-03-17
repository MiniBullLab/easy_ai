#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.pose2d.pose2d_dataset_process import Pose2dDataSetProcess


class Box2dLoader(DataLoader):

    def __init__(self, box_list, src_image, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(data_channel)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.dataset_process = Pose2dDataSetProcess(resize_type, normalize_type,
                                                    mean, std, self.get_pad_color())
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
        image, box = self.dataset_process.expand_dataset(self.src_image, box, self.expand_ratio)
        image = self.dataset_process.resize_image(image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        return box, torch_image

    def __len__(self):
        return self.count
