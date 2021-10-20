#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import time
import torch
from easyai.utility.env import set_random_seed
from easyai.utility.logger import EasyLogger


class TorchDeviceProcess():

    first_run = True
    cuda = None

    def __init__(self):
        self.device = torch.device("cpu")

    @classmethod
    def hasCUDA(cls):
        cls.cuda = torch.cuda.is_available()
        if cls.cuda:
            return True
        else:
            return False

    def setGpuId(self, id=0):
        if TorchDeviceProcess.cuda:
            count = self.getCUDACount()
            if id >= 0 and id < count:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
            else:
                EasyLogger.error("GPU %d error" % id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        EasyLogger.debug("Using device: \"{}\"".format(self.device))

    def getCUDACount(self):
        count = -1
        if TorchDeviceProcess.cuda:
            count = torch.cuda.device_count()
        return count

    def initTorch(self):
        if TorchDeviceProcess.first_run:
            torch.cuda.empty_cache()
            t = time.time()
            set_random_seed(int(t), TorchDeviceProcess.hasCUDA())
            TorchDeviceProcess.first_run = False
