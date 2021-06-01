#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Landmark)
class LandmarkShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Landmark)

    def show(self, src_image, objects_landmark, skeleton):
        image = src_image.copy()
        self.drawing.draw_keypoint2d_result(image, objects_landmark, skeleton)
        self.drawing.draw_image("image", image, 0.8)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
