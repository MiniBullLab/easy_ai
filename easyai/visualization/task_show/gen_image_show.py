#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.config.name_manager import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.GenerateImage)
class GenerateImage(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.GenerateImage)

    def show(self, src_image, scale=1.0):
        image = src_image.copy()
        self.drawing.draw_image('image', image, scale)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()