#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import abc
from easyai.helper.image_process import ImageProcess
from easyai.utility.logger import EasyLogger


class BaseDataGeter():

    def __init__(self, data_channel, transform_func=None):
        self.data_channel = data_channel
        self.transform_func = transform_func
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

    def read_src_image(self, image_path):
        src_image = None
        cv_image = None
        if self.image_process.isImageFile(image_path):
            if self.data_channel == 1:
                src_image = self.image_process.read_gray_image(image_path)
                if src_image is not None:
                    cv_image = src_image[:]
            elif self.data_channel == 3:
                cv_image, src_image = self.image_process.readRgbImage(image_path)
            else:
                EasyLogger.error("data channel not support(%d)!" % self.data_channel)
        else:
            EasyLogger.error("%s not image" % image_path)
        assert src_image is not None, EasyLogger.error("read %s error!" % image_path)
        return cv_image, src_image

    def get_pad_color(self):
        result = None
        if self.data_channel == 1:
            result = 0
        elif self.data_channel == 3:
            result = (0, 0, 0)
        else:
            EasyLogger.error("data channel(%d) not support!" % self.data_channel)
        return result

    @abc.abstractmethod
    def get(self, **kwargs):
        pass

