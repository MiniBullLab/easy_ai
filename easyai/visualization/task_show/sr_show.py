#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.base_name.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.SuperResolution_Task)
class SuperResolutionShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.SuperResolution_Task)

    def show(self, src_image, sr_image, scale=0.5):
        self.drawing.draw_image("src_image", src_image, scale)
        self.drawing.draw_image("sr_image", sr_image, scale)
        return self.wait_key()
