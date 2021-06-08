#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess


class TaskPostProcess():

    def __init__(self):
        self.process_func = None
        self.dataset_process = ImageDataSetProcess()

    def set_threshold(self, value):
        assert self.process_func is not None
        self.process_func.set_threshold(value)

    @abc.abstractmethod
    def post_process(self, *args, **kwargs):
        pass
