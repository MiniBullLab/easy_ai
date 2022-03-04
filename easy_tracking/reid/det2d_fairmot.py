#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.name_manager.task_name import TaskName
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.config.utility.base_config import BaseConfig
from easyai.config.utility.config_factory import ConfigFactory
from easyai.data_loader.common.numpy_data_geter import NumpyDataGeter
from easy_tracking.fairmot.fairmot_post_process import FairMOTPostProcess
from easyai.utility.logger import EasyLogger


class Det2dFairMOT():

    def __init__(self, model_name, gpu_id, weights_path, config_path):
        self.config_factory = ConfigFactory()
        self.config_path = config_path
        self.task_config = None
        self.model_args = {"type": model_name[0],
                           "data_channel": 3,
                           "class_number": 1,
                           "reid": 64}
        self.model_process = TorchModelProcess()
        self.model = self.model_process.create_model(self.model_args, gpu_id)
        self.model_process.load_latest_model(weights_path[0], self.model)
        self.model = self.model_process.model_test_init(self.model)
        self.model.eval()
        self.device = self.model_process.get_device()

        self.image_size = (1088, 608)
        self.data_geter = NumpyDataGeter(self.image_size, 3, 2, 1,)

        self.post_process = FairMOTPostProcess(self.image_size, 1)

    def process(self, src_image):
        input_data = self.data_geter.get(src_image)
        with torch.no_grad():
            img_batch = input_data['image'].to(self.device)
            shape = input_data['src_image'].shape[:2]  # shape = [height, width]
            src_size = (shape[1], shape[0])
            output_list = self.model(img_batch)
            result = self.post_process(output_list, src_size)
        return result

    def set_task_config(self, config=None):
        if config is None:
            self.task_config = self.config_factory.get_config(TaskName.DET2D_REID_TASK,
                                                              self.config_path)
            self.task_config.save_config()
        elif isinstance(config, str):
            self.task_config = self.config_factory.get_config(TaskName.DET2D_REID_TASK,
                                                              self.config_path)
            self.task_config.save_config()
        elif isinstance(config, BaseConfig):
            self.config_path = None
            self.task_config = config
        assert self.task_config is not None, \
            EasyLogger.error("create config fail! {}".format(config))

    def set_model(self, my_model=None, gpu_id=0):
        if my_model is None:
            self.model = self.model_process.create_model(self.model_args, gpu_id)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.eval()
        assert self.model is not None, EasyLogger.error("create model fail!")
