#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import numpy as np
import torch
from easyai.data_loader.common.numpy_data_geter import NumpyDataGeter
from easyai.helper.timer_process import TimerProcess
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.config.utility.base_config import BaseConfig
from easyai.config.utility.config_factory import ConfigFactory
from easyai.utility.logger import EasyLogger


class ReidInference():

    def __init__(self, model_name, config_path, task_name):
        self.task_name = task_name
        self.config_path = config_path
        self.config_factory = ConfigFactory()
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.model_name = model_name
        self.model_args = None
        self.model = None
        self.task_config = None
        self.src_size = (0, 0)
        self.set_task_config(config_path)

    def set_task_config(self, config=None):
        if config is None:
            self.task_config = self.config_factory.get_config(self.task_name, self.config_path)
            self.task_config.save_config()
        elif isinstance(config, str):
            self.task_config = self.config_factory.get_config(self.task_name, self.config_path)
            self.task_config.save_config()
        elif isinstance(config, BaseConfig):
            self.config_path = None
            self.task_config = config
        assert self.task_config is not None, \
            EasyLogger.error("create config fail! {}".format(config))

    def set_model_param(self, data_channel, **params):
        if self.model_name is None or len(self.model_name) == 0:
            if self.task_config.model_config is not None:
                self.model_args = self.task_config.model_config
            else:
                EasyLogger.error("{} : model config error!".format(self.model_args))
        else:
            if isinstance(self.model_name, (list, tuple)):
                if len(self.model_name) > 0:
                    self.model_args = {"type": self.model_name[0]}
                else:
                    self.model_args = {"type": None}
            elif isinstance(self.model_name, str):
                self.model_args = {"type": self.model_name}
            else:
                pass
            self.model_args["data_channel"] = data_channel
            self.model_args.update(params)
        EasyLogger.debug(self.model_args)

    def set_model(self, my_model=None, gpu_id=0):
        if my_model is None:
            self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.eval()
        assert self.model is not None, EasyLogger.error("create model fail!")

    @abc.abstractmethod
    def process(self, input_path, data_type=1, is_show=False):
        pass

    @abc.abstractmethod
    def single_image_process(self, input_data):
        pass

    @abc.abstractmethod
    def infer(self, input_data, net_type=0):
        pass

    def load_weights(self, weights_path):
        if isinstance(weights_path, (list, tuple)):
            self.torchModelProcess.load_latest_model(weights_path[0], self.model)
        else:
            self.torchModelProcess.load_latest_model(weights_path, self.model)
        self.model = self.torchModelProcess.model_test_init(self.model)
        self.model.eval()

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name

    def get_single_image_data(self, input_param):
        EasyLogger.debug(self.task_config.data)
        image_size = tuple(self.task_config.data['image_size'])
        data_channel = self.task_config.data['data_channel']
        mean = self.task_config.data.get('mean', 1)
        std = self.task_config.data.get('std', 0)
        resize_type = self.task_config.data['resize_type']
        normalize_type = self.task_config.data['normalize_type']
        if isinstance(input_param, np.ndarray):
            data_geter = NumpyDataGeter(image_size, data_channel,
                                        resize_type, normalize_type, mean, std)
            input_data = data_geter.get(input_param)
            return input_data
        else:
            EasyLogger.debug("input path not support!")
            return None

    def common_output(self, output_list):
        output = None
        count = len(output_list)
        loss_count = len(self.model.lossList)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            output = self.model.lossList[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            output = self.model.lossList[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            output = []
            for i in range(0, count):
                temp = self.model.lossList[i](output_list[i])
                output.append(temp)
        else:
            EasyLogger.error("compute prediction error")
        return output

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])

    @property
    def device(self):
        return self.torchModelProcess.get_device()
