#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import abc
from easyai.helper.timer_process import TimerProcess
from easyai.tasks.utility.base_task import BaseTask


class BaseTest(BaseTask):

    def __init__(self, config_path, task_name):
        super().__init__()
        self.set_task_name(task_name)
        self.timer = TimerProcess()
        self.config_path = config_path
        self.test_task_config = self.config_factory.get_config(self.task_name, self.config_path)

    @abc.abstractmethod
    def load_weights(self, weights_path):
        pass

    @abc.abstractmethod
    def test(self, val_path):
        pass
