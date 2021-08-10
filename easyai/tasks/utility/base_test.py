#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.helper.average_meter import AverageMeter
from easyai.data_loader.utility.dataloader_factory import DataloaderFactory
from easyai.tasks.utility.batch_data_process_factory import BatchDataProcessFactory
from easyai.evaluation.utility.evaluation_factory import EvaluationFactory
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask
from easyai.utility.logger import EasyLogger


class BaseTest(BaseTask):

    def __init__(self, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.epoch_loss_average = AverageMeter()
        self.dataloader_factory = DataloaderFactory()
        self.batch_data_process_factory = BatchDataProcessFactory()
        self.evaluation_factory = EvaluationFactory()
        self.evaluation_args = None
        self.test_task_config = None
        self.dataloader = None
        self.batch_data_process_func = None
        self.total_batch_data = 0
        self.inference = None
        self.model = None
        self.device = None
        self.evaluation = None
        self.val_path = None

    def set_test_config(self, config=None):
        if isinstance(config, BaseConfig):
            self.test_task_config = config
        assert self.test_task_config is not None, \
            EasyLogger.error("set config fail！{}".format(config))

    def set_model(self, my_model=None):
        if my_model is None:
            self.model = self.inference.model
            self.device = self.inference.device
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.device = my_model.device

    def start_test(self):
        if self.evaluation is not None:
            self.evaluation.reset()
        self.epoch_loss_average.reset()
        self.model.eval()
        self.timer.tic()
        EasyLogger.warn("Test data count is : %d" % self.total_batch_data)
        return self.total_batch_data > 0

    def metirc_loss(self, step, loss_value):
        if loss_value != float("inf") and loss_value != float("nan"):
            self.epoch_loss_average.update(loss_value)

    def print_test_info(self, step, loss_value=-1):
        if loss_value >= 0:
            info_str = "Val Batch [{}/{}] loss: {:.7f} | Time: {:.5f}".format(step + 1,
                                                                              self.total_batch_data,
                                                                              loss_value,
                                                                              self.timer.toc(True))
        else:
            info_str = "Val Batch [{}/{}] | Time: {:.5f}".format(step + 1,
                                                                 self.total_batch_data,
                                                                 self.timer.toc(True))
        EasyLogger.info(info_str)
        print(info_str)

    def load_weights(self, weights_path):
        if self.inference is not None:
            self.inference.load_weights(weights_path)
        else:
            EasyLogger.error("inference init error")

    @abc.abstractmethod
    def process_test(self, val_path, epoch=0):
        pass

    @abc.abstractmethod
    def test(self, epoch=0):
        pass

    def create_dataloader(self, data_path):
        assert self.test_task_config is not None
        dataloader_config = self.test_task_config.val_data.get('dataloader', None)
        dataset_config = self.test_task_config.val_data.get('dataset', None)
        self.dataloader = self.dataloader_factory.get_val_dataloader(data_path,
                                                                     dataloader_config,
                                                                     dataset_config)
        self.batch_data_process_func = \
            self.batch_data_process_factory.build_process(self.test_task_config.batch_data_process)
        if self.dataloader is not None:
            self.total_batch_data = len(self.dataloader)
        else:
            self.total_batch_data = 0
        self.val_path = data_path

    def compute_loss(self, model_output, batch_data):
        loss = 0
        if self.test_task_config.model_type == 0:
            loss = self.common_loss(model_output, batch_data)
        elif self.test_task_config.model_type == 1:
            loss = self.gan_loss(model_output, batch_data)
        return loss

    def common_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.lossList[0](output_list[0], batch_data)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.lossList[0](output_list, batch_data)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.lossList[k](output_list[k], batch_data)
            else:
                EasyLogger.error("compute loss error")
        return loss.item()

    def gan_loss(self, output_list, batch_data):
        loss = 0
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        with torch.no_grad():
            if loss_count == 1 and output_count == 1:
                loss = self.model.g_loss_list[0](output_list[0], batch_data)
            elif loss_count == 1 and output_count > 1:
                loss = self.model.g_loss_list[0](output_list, batch_data)
            elif loss_count > 1 and loss_count == output_count:
                for k in range(0, loss_count):
                    loss += self.model.g_loss_list[k](output_list[k], batch_data)
            else:
                EasyLogger.error("compute gan loss error")
        return loss.item()

    def batch_processing(self, batch_data):
        if self.batch_data_process_func is not None:
            self.batch_data_process_func(batch_data)
