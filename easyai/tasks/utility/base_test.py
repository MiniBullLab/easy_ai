#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.helper.average_meter import AverageMeter
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask


class BaseTest(BaseTask):

    def __init__(self, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.epoch_loss_average = AverageMeter()
        self.test_task_config = None
        self.inference = None
        self.model = None
        self.device = None

    def set_test_config(self, config=None):
        if isinstance(config, BaseConfig):
            self.test_task_config = config

    def set_model(self, my_model=None):
        if my_model is None:
            self.model = self.inference.model
            self.device = self.inference.device
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.device = my_model.device
            self.model.eval()

    @abc.abstractmethod
    def load_weights(self, weights_path):
        pass

    @abc.abstractmethod
    def test(self, val_path, epoch=0):
        pass
