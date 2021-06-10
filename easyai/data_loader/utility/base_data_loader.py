#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch
import abc
from easyai.helper.image_process import ImageProcess


class DataLoader():

    def __init__(self, data_channel):
        self.data_channel = data_channel
        self.image_process = ImageProcess()

    def all_numpy_to_tensor(self, input_data):
        result = None
        if input_data is None:
            result = None
        elif input_data.ndim == 3:
            result = torch.from_numpy(input_data).unsqueeze(0)
        elif input_data.ndim == 4:
            result = torch.from_numpy(input_data)
        return result

    def expand_dim(self, input_data):
        result = None
        if input_data is None:
            result = None
        elif input_data.ndim == 3:
            result = input_data.unsqueeze(0)
        elif input_data.ndim == 4:
            result = input_data
        return result

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def __iter__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.data_channel == 1:
            src_image = self.image_process.read_gray_image(image_path)
            cv_image = src_image[:]
        elif self.data_channel == 3:
            cv_image, src_image = self.image_process.readRgbImage(image_path)
        else:
            print("dataloader src image error!")
        return cv_image, src_image

    def get_pad_color(self):
        result = None
        if self.data_channel == 1:
            result = 0
        elif self.data_channel == 3:
            result = (0, 0, 0)
        return result

