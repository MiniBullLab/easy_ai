#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess


class BasePostProcess():

    def __init__(self):
        self.threshold = 0
        self.dataset_process = ImageDataSetProcess()

    def set_threshold(self, value):
        self.threshold = value
