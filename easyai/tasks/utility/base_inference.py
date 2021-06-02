#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import abc
from pathlib import Path
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.common.images_loader import ImagesLoader
from easyai.data_loader.common.video_loader import VideoLoader
from easyai.data_loader.common.text_data_loader import TextDataLoader
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.visualization.utility.task_show_factory import TaskShowFactory
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask


class BaseInference(BaseTask):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(config_path)
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.show_factory = TaskShowFactory()
        self.model = None
        self.task_config = None
        self.src_size = (0, 0)
        self.result_show = self.show_factory.get_task_show(self.task_name)
        if isinstance(model_name, (list, tuple)):
            if len(model_name) > 0:
                self.model_args = {"type": model_name[0]}
            else:
                self.model_args = {"type": None}
        elif isinstance(model_name, str):
            self.model_args = {"type": model_name}

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

    def set_model_param(self, data_channel, **params):
        self.model_args["data_channel"] = data_channel
        self.model_args.update(params)

    def set_model(self, my_model=None, gpu_id=0):
        if my_model is None:
            self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.eval()

    @abc.abstractmethod
    def process(self, input_path, data_type=1, is_show=False):
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
        image_size = self.task_config.image_size
        data_channel = self.task_config.data_channel
        mean = self.task_config.data_mean
        std = self.task_config.data_std
        normalize_type = self.task_config.normalize_type
        resize_type = self.task_config.resize_type
        if Path(input_path).is_dir():
            dataloader = ImagesLoader(input_path, image_size, data_channel,
                                      resize_type, normalize_type, mean, std)
        elif Path(input_path).suffix in ['.txt', '.text']:
            dataloader = TextDataLoader(input_path, image_size, data_channel,
                                        resize_type, normalize_type, mean, std)
        else:
            dataloader = VideoLoader(input_path, image_size, data_channel,
                                     resize_type, normalize_type, mean, std)
        return dataloader

    def set_src_size(self, src_data):
        shape = src_data.shape[:2]  # shape = [height, width]
        self.src_size = (shape[1], shape[0])

    @property
    def device(self):
        return self.torchModelProcess.get_device()
