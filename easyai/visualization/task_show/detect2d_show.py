#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Detect2d_Task)
class DetectionShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Detect2d_Task)

    def show(self, src_image, detection_objects):
        image = src_image.copy()
        self.drawing.draw_detect_objects(image, detection_objects)
        self.drawing.draw_image("image", image, 0.8)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
