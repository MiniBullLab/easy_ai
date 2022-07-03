#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from torch.utils.data import Dataset
from easyai.helper.image_process import ImageProcess
from easyai.utility.logger import EasyLogger


class TorchDataLoader(Dataset):

    def __init__(self, data_path, data_channel, transform_func=None):
        super().__init__()
        self.data_path = data_path
        self.data_channel = data_channel
        self.transform_func = transform_func
        self.image_process = ImageProcess()

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

