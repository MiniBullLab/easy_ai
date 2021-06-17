#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.helper.average_meter import AverageMeter
from easyai.data_loader.utility.dataloader_factory import DataloaderFactory
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask


class BaseTest(BaseTask):

    def __init__(self, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.epoch_loss_average = AverageMeter()
        self.dataloader_factory = DataloaderFactory()
        self.test_task_config = None
        self.dataloader = None
        self.total_batch_image = 0
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

    def start_test(self):
        self.epoch_loss_average.reset()
        self.model.eval()
        self.timer.tic()
        assert self.total_batch_image > 0

    def metirc_loss(self, step, loss_value):
        self.epoch_loss_average.update(loss_value)
        print("Val Batch {} loss: {:.7f} | Time: {:.5f}".format(step,
                                                                loss_value,
                                                                self.timer.toc(True)))

    @abc.abstractmethod
    def load_weights(self, weights_path):
        pass

    @abc.abstractmethod
    def test(self, val_path, epoch=0):
        pass

    def create_dataloader(self, data_path):
        assert self.test_task_config is not None
        dataloader_config = self.test_task_config['dataloader']
        dataset_config = self.test_task_config['dataset']
        self.dataloader = self.dataloader_factory.get_val_dataloader(data_path,
                                                                     dataloader_config,
                                                                     dataset_config)
        if self.dataloader is not None:
            self.total_batch_image = len(self.dataloader)
        else:
            self.total_batch_image = 0
