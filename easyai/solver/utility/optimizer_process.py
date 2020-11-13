#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.utility.registry import build_from_cfg
from easyai.solver.utility.registry import REGISTERED_OPTIMIZER


class OptimizerProcess():

    def __init__(self, base_lr):
        self.base_lr = base_lr
        self.optimizer = None

    def get_optimizer_config(self, epoch, config):
        em = 0
        for e in config.keys():
            if epoch >= e:
                em = e
        config_args = config[em]
        return config_args

    def get_optimizer(self, config, model):
        config_args = config.copy()
        params = filter(lambda p: p.requires_grad, model.parameters())
        config_args['params'] = params
        config_args['lr'] = self.base_lr
        self.optimizer = build_from_cfg(config, REGISTERED_OPTIMIZER)
        self.print_param()
        return self.optimizer

    def adjust_optimizer(self, config):
        config_args = config.copy()
        config_args['params'] = None
        config_args['lr'] = self.base_lr
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in config_args:
                    config_args['key'] = param_group[key]
        self.optimizer = build_from_cfg(config, REGISTERED_OPTIMIZER)
        return self.optimizer

    def print_param(self):
        if self.optimizer is None:
            return
        for i_group, param_group in enumerate(self.optimizer.param_groups):
            for key in param_group.keys():
                print('OPTIMIZER - group %s setting %s = %s' %
                      (i_group, key, param_group[key]))

    def adjust_param(self, optimizer, setting):
        for i_group, param_group in enumerate(optimizer.param_groups):
            for key in param_group.keys():
                if key in setting:
                    param_group[key] = setting[key]
                    print('OPTIMIZER - group %s setting %s = %s' %
                          (i_group, key, param_group[key]))



