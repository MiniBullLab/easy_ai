#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import torch
from easyai.helper.timer_process import TimerProcess
from easyai.data_loader.utility.dataloader_factory import DataloaderFactory
from easyai.tasks.utility.batch_data_process_factory import BatchDataProcessFactory
from easyai.torch_utility.torch_model_process import TorchModelProcess
from easyai.solver.utility.optimizer_process import OptimizerProcess
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.solver.utility.freeze_process import FreezePorcess
from easyai.torch_utility.train_log import TrainLogger
from easyai.config.utility.base_config import BaseConfig
from easyai.tasks.utility.base_task import BaseTask
from easyai.utility.logger import EasyLogger


class BaseTrain(BaseTask):

    def __init__(self, model_name, config_path, task_name):
        super().__init__(config_path)
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.dataloader_factory = DataloaderFactory()
        self.batch_data_process_factory = BatchDataProcessFactory()
        self.torchModelProcess = TorchModelProcess()
        self.freeze_process = FreezePorcess()
        self.model_name = model_name
        self.model_args = None
        self.model = None
        self.train_task_config = None
        self.batch_data_process_func = None
        self.dataloader = None
        self.total_batch_data = 0
        self.is_sparse = False
        self.sparse_ratio = 0.0

        self.set_train_config(config_path)

        self.optimizer_process = OptimizerProcess(base_lr=self.train_task_config.base_lr)
        self.lr_factory = LrSchedulerFactory(self.train_task_config.base_lr,
                                             self.train_task_config.max_epochs)
        self.train_logger = TrainLogger(self.train_task_config.log_name,
                                        EasyLogger.ROOT_DIR)

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
        if self.model_name is None or len(self.model_name) == 0:
            if self.train_task_config.model_config is not None:
                self.model_args = self.train_task_config.model_config
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
                EasyLogger.error("model config error!(%s)" % self.model_name)
            self.model_args["data_channel"] = data_channel
            self.model_args.update(params)
        EasyLogger.debug(self.model_args)

    def set_model(self, my_model=None, gpu_id=0, init_type="kaiming"):
        if my_model is None:
            EasyLogger.debug(self.model_args)
            self.model = self.torchModelProcess.create_model(self.model_args, gpu_id)
            self.torchModelProcess.init_model(self.model, init_type)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.train()
        assert self.model is not None, EasyLogger.error("create model fail!")

    @abc.abstractmethod
    def load_pretrain_model(self, weights_path):
        pass

    @abc.abstractmethod
    def train(self, train_path, val_path):
        pass

    @abc.abstractmethod
    def compute_backward(self, batch_data, step_index):
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

    def create_dataloader(self, data_path):
        assert self.train_task_config is not None
        dataloader_config = self.train_task_config.train_data.get('dataloader', None)
        dataset_config = self.train_task_config.train_data.get('dataset', None)
        self.dataloader = self.dataloader_factory.get_train_dataloader(data_path,
                                                                       dataloader_config,
                                                                       dataset_config)
        self.batch_data_process_func = \
            self.batch_data_process_factory.build_process(self.train_task_config.batch_data_process)
        if self.dataloader is not None:
            self.total_batch_data = len(self.dataloader)
        else:
            self.total_batch_data = 0

    def batch_processing(self, batch_data):
        if self.batch_data_process_func is not None:
            self.batch_data_process_func(batch_data)



