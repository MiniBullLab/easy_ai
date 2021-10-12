#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.data_loader.utility.base_data_geter import BaseDataGeter
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess
from easyai.utility.logger import EasyLogger


class NumpyDataGeter(BaseDataGeter):

    def __init__(self, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1, transform_func=None):
        super().__init__(data_channel, transform_func)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.dirProcess = DirProcess()
        self.dataset_process = TaskDataSetProcess(resize_type, normalize_type,
                                                  mean, std,
                                                  pad_color=self.get_pad_color())

    def get(self, numpy_image):
        cv_image, src_image = self.get_src_image(numpy_image)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        if self.transform_func is not None:
            torch_image = self.transform_func(torch_image)
        torch_image = torch_image.unsqueeze(0)
        return {"src_image": cv_image,
                "image": torch_image}

    def get_src_image(self, numpy_image):
        src_image = None
        cv_image = None
        if numpy_image is not None:
            if self.data_channel == 1:
                src_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)
                if src_image is not None:
                    cv_image = numpy_image[:]
            elif self.data_channel == 3:
                src_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
                cv_image = numpy_image[:]
            else:
                EasyLogger.error("data channel not support(%d)!" % self.data_channel)
        else:
            EasyLogger.error("image is None")
        assert src_image is not None, EasyLogger.error("get image error!")
        return cv_image, src_image
