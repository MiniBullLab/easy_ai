#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import abc
from pathlib import Path

import numpy as np
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.helper import VideoProcess
from easyai.data_loader.common.images_loader import ImagesLoader
from easyai.data_loader.common.video_loader import VideoLoader
from easyai.data_loader.common.text_data_loader import TextDataLoader
from easyai.data_loader.common.numpy_data_geter import NumpyDataGeter
from easyai.data_loader.utility.data_transforms_factory import DataTransformsFactory
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.tasks.utility.batch_data_process_factory import BatchDataProcessFactory
from easyai.visualization.utility.task_show_factory import TaskShowFactory
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask
from easyai.utility.logger import EasyLogger


class BaseInference(BaseTask):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(config_path)
        self.set_task_name(task_name)
        self.video_process = VideoProcess()
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.show_factory = TaskShowFactory()
        self.transform_factory = DataTransformsFactory()
        self.batch_data_process_factory = BatchDataProcessFactory()
        self.model_name = model_name
        self.model_args = None
        self.model = None
        self.task_config = None
        self.batch_data_process_func = None
        self.src_size = (0, 0)
        self.result_show = self.show_factory.get_task_show(self.task_name)
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

    def get_image_data_lodaer(self, input_path):
        if not os.path.exists(input_path):
            return None
        EasyLogger.debug(self.task_config.data)
        image_size = tuple(self.task_config.data['image_size'])
        data_channel = self.task_config.data['data_channel']
        mean = self.task_config.data.get('mean', 1)
        std = self.task_config.data.get('std', 0)
        resize_type = self.task_config.data['resize_type']
        normalize_type = self.task_config.data['normalize_type']
        transform_args = self.task_config.data.get('transform_func', None)
        transform_func = self.transform_factory.get_data_transform(transform_args)
        if Path(input_path).is_dir():
            dataloader = ImagesLoader(input_path, image_size, data_channel,
                                      resize_type, normalize_type, mean, std,
                                      transform_func)
        elif Path(input_path).suffix in ['.txt', '.text']:
            dataloader = TextDataLoader(input_path, image_size, data_channel,
                                        resize_type, normalize_type, mean, std,
                                        transform_func)
        elif self.video_process.isVideoFile(input_path):
            dataloader = VideoLoader(input_path, image_size, data_channel,
                                     resize_type, normalize_type, mean, std,
                                     transform_func)
        else:
            EasyLogger.debug("%s: input path not support!" % input_path)
            return None
        self.batch_data_process_func = \
            self.batch_data_process_factory.build_process(self.task_config.batch_data_process)
        EasyLogger.debug("data count: %d" % len(dataloader))
        return dataloader

    def get_point_cloud_data_lodaer(self, input_path):
        pass

    def get_single_image_data(self, input_param):
        EasyLogger.debug(self.task_config.data)
        image_size = tuple(self.task_config.data['image_size'])
        data_channel = self.task_config.data['data_channel']
        mean = self.task_config.data.get('mean', 1)
        std = self.task_config.data.get('std', 0)
        resize_type = self.task_config.data['resize_type']
        normalize_type = self.task_config.data['normalize_type']
        transform_args = self.task_config.data.get('transform_func', None)
        transform_func = self.transform_factory.get_data_transform(transform_args)
        if isinstance(input_param, np.ndarray):
            data_geter = NumpyDataGeter(image_size, data_channel,
                                        resize_type, normalize_type, mean, std,
                                        transform_func)
            input_data = data_geter.get(input_param)
            return input_data
        else:
            EasyLogger.debug("input path not support!")
            return None

    def batch_processing(self, batch_data):
        if self.batch_data_process_func is not None:
            self.batch_data_process_func(batch_data)

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])

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

    def gan_output(self, output_list):
        output = None
        loss_count = len(self.model.g_loss_list)
        output_count = len(output_list)
        if loss_count == 1 and output_count == 1:
            output = self.model.g_loss_list[0](output_list[0])
        elif loss_count == 1 and output_count > 1:
            output = self.model.g_loss_list[0](output_list)
        elif loss_count > 1 and loss_count == output_count:
            output = []
            for k in range(0, loss_count):
                result = self.model.g_loss_list[k](output_list[k])
                output.append(result)
        else:
            EasyLogger.error("compute generator prediction error")
        return output

    @property
    def device(self):
        return self.torchModelProcess.get_device()
