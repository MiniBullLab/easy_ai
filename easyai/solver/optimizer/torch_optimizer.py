#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch
from easyai.config.name_manager import OptimizerName
from easyai.solver.utility.registry import REGISTERED_OPTIMIZER

REGISTERED_OPTIMIZER.add_module(torch.optim.SGD, OptimizerName.SGD)
REGISTERED_OPTIMIZER.add_module(torch.optim.ASGD, OptimizerName.ASGD)
REGISTERED_OPTIMIZER.add_module(torch.optim.Adam, OptimizerName.Adam)
REGISTERED_OPTIMIZER.add_module(torch.optim.Adamax, OptimizerName.Adamax)
REGISTERED_OPTIMIZER.add_module(torch.optim.Adagrad, OptimizerName.Adagrad)
REGISTERED_OPTIMIZER.add_module(torch.optim.Rprop, OptimizerName.Rprop)
REGISTERED_OPTIMIZER.add_module(torch.optim.RMSprop, OptimizerName.RMSprop)
