#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.PC_Classify_Task)
class PointCloudClassifyShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.PC_Classify_Task)

    def show(self):
        return False
