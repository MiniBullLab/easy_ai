#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import torch
from easyai.utility.logger import EasyLogger
from easyai.tasks.utility.common_train import CommonTrain

from easy_pc.model.utility.pc_model_factory import PCModelFactory


class BasePCTrain(CommonTrain):

    def __init__(self,  model_name, config_path, task_name):
        super().__init__(model_name, config_path, task_name)

    def set_model(self, my_model=None, gpu_id=0, init_type="kaiming"):
        if my_model is None:
            EasyLogger.debug(self.model_args)
            model_factory = PCModelFactory()
            self.model = self.torchModelProcess.create_model(self.model_args,
                                                             model_factory,
                                                             gpu_id)
            self.torchModelProcess.init_model(self.model, init_type)
        elif isinstance(my_model, torch.nn.Module):
            self.model = my_model
            self.model.train()
        assert self.model is not None, EasyLogger.error("create model fail!")

