#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.RecognizeText)
class RecognizeTextShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.RecognizeText)

    def show(self, src_image, scale=1.0):
        self.drawing.draw_image("src_image", src_image, scale)
        return self.wait_key()
