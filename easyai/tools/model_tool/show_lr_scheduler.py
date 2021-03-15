#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import torch.nn as nn
import matplotlib.pyplot as plt
from easyai.solver.utility.optimizer_process import OptimizerProcess
from easyai.solver.utility.lr_factory import LrSchedulerFactory
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class ShowLrScheduler():

    def __init__(self, task_name, config_path=None, epoch_iteration=4000):
        self.epoch_iteration = epoch_iteration
        self.config_factory = ConfigFactory()
        self.task_config = self.config_factory.get_config(task_name, config_path)
        self.lr_factory = LrSchedulerFactory(self.task_config.base_lr,
                                             self.task_config.max_epochs,
                                             self.epoch_iteration)
        self.lr_scheduler = self.lr_factory.get_lr_scheduler(self.task_config.lr_scheduler_config)
        self.optimizer_process = OptimizerProcess(base_lr=self.task_config.base_lr)

    def show(self):
        lr_list = self.get_lr_lists()
        lr_class_name = self.task_config.lr_scheduler_config['lr_type'].strip()
        self.show_graph(lr_list, self.task_config.max_epochs, self.epoch_iteration, lr_class_name)

    def get_lr_lists(self):
        lr_list = []
        model = nn.Linear(1280, 1000)
        optimizer_args = self.optimizer_process.get_optimizer_config(0, self.task_config.optimizer_config)
        optimizer = self.optimizer_process.get_optimizer(optimizer_args,
                                                         model)
        for epoch in range(0, self.task_config.max_epochs):
            for idx in range(0, self.epoch_iteration):
                current_iter = epoch * self.epoch_iteration + idx
                lr = self.lr_scheduler.get_lr(epoch, current_iter)
                self.lr_scheduler.adjust_learning_rate(optimizer, lr)
                lr_list.append(optimizer.param_groups[0]['lr'])
        return lr_list

    def show_graph(self, lr_lists, epochs, steps, lr_name):
        plt.clf()
        plt.rcParams['figure.figsize'] = [20, 5]
        x = list(range(epochs * steps))
        plt.plot(x, lr_lists, label=lr_name)
        plt.plot()

        plt.ylim(10e-7, 1)
        plt.yscale("log")
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.title("plot learning rate secheduler {}".format(lr_name))
        plt.legend()
        plt.show()


def main(options_param):
    print("process start...")
    show_lr = ShowLrScheduler(options_param.task_name,
                              options_param.config_path,
                              options_param.epoch_iteration)
    show_lr.show()
    print("process end!")


if __name__ == '__main__':
    options = ToolArgumentsParse.show_lr_parse()
    main(options)
