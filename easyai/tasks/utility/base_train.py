#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.solver.utility.optimizer_process import OptimizerProcess
from easyai.solver.utility.freeze_process import FreezePorcess
from easyai.torch_utility.train_log import TrainLogger
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask


class BaseTrain(BaseTask):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(config_path)
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.torchModelProcess = TorchModelProcess()
        self.freeze_process = FreezePorcess()
        self.model = None
        self.train_task_config = None
        self.is_sparse = False
        self.sparse_ratio = 0.0

        if isinstance(model_name, (list, tuple)):
            self.model_args = {"type": model_name[0]}
        elif isinstance(model_name, str):
            self.model_args = {"type": model_name}

        self.set_train_config(config_path)

        self.optimizer_process = OptimizerProcess(base_lr=self.train_task_config.base_lr)
        self.train_logger = TrainLogger(self.train_task_config.log_name,
                                        self.train_task_config.root_save_dir)

    def set_train_config(self, config=None):
        if config is None:
            self.train_task_config = self.config_factory.get_config(self.task_name, self.config_path)
            self.train_task_config.save_config()
        elif isinstance(config, str):
            self.train_task_config = self.config_factory.get_config(self.task_name, self.config_path)
            self.train_task_config.save_config()
        elif isinstance(config, BaseConfig):
            self.config_path = None
            self.train_task_config = config

    def set_model_param(self, data_channel, **params):
        self.model_args["data_channel"] = data_channel
        self.model_args.update(params)

    def set_model(self, my_model=None, gpu_id=0):
        if my_model is None:
            self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.train()

    @abc.abstractmethod
    def load_latest_param(self, latest_weights_path):
        pass

    @abc.abstractmethod
    def train(self, train_path, val_path):
        pass

    @abc.abstractmethod
    def compute_backward(self, input_datas, targets, step_index):
        pass

    @abc.abstractmethod
    def compute_loss(self, output_list, targets, loss_type=0):
        pass

    def set_is_sparse_train(self, is_sparse=False, sparse_ratio=0.0):
        self.is_sparse = is_sparse
        self.sparse_ratio = sparse_ratio

    def print_grad_norm(self):
        if self.model is not None:
            for p in self.model.parameters():
                print(p.grad.norm())

    @property
    def device(self):
        return self.torchModelProcess.get_device()

