#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.registry import build_from_cfg
from easyai.name_manager.solver_name import OptimizerName
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
        optimizer_type = config['type']
        if optimizer_type.strip() == OptimizerName.LARCOptimizer:
            config_args = config.copy()
            optimizer_args = config_args.pop('optimizer_args')
            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer_args['params'] = params
            optimizer_args['lr'] = self.base_lr
            optim = build_from_cfg(optimizer_args, REGISTERED_OPTIMIZER)
            config_args['optimizer'] = optim
            self.optimizer = build_from_cfg(config_args, REGISTERED_OPTIMIZER)
        else:
            config_args = config.copy()
            params = filter(lambda p: p.requires_grad, model.parameters())
            config_args['params'] = params
            config_args['lr'] = self.base_lr
            self.optimizer = build_from_cfg(config_args, REGISTERED_OPTIMIZER)
            # self.print_param()
        return self.optimizer

    def bias_not_weight_decay(self, config, model):
        config_args = config.copy()
        spe_params = []
        conv_params = []
        for k, v in model.named_parameters():
            if 'bn' in k or 'bias' in k:
                spe_params.append(v)
            else:
                conv_params.append(v)
        params_group = [{'params': spe_params, 'weight_decay': 0.0}, {'params': conv_params}]
        config_args['params'] = params_group
        config_args['lr'] = self.base_lr
        self.optimizer = build_from_cfg(config_args, REGISTERED_OPTIMIZER)
        # self.print_param()
        return self.optimizer

    def get_optim_params(self, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_lr
            weight_decay = 0.0001
            if "bias" in key:
                lr = self.base_lr * 2
                weight_decay = 0.0001
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        return params

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



