#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.base_name.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Classify_Task)
class ClassifyShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Classify_Task)

    def show(self, src_image, result,
             class_name, scale=1.0):
        image = src_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_name[result],
                    (int(image.shape[0] * 0.1), int(image.shape[1] * 0.1)),
                    font, 0.5, (0, 255, 0), 2)

        self.drawing.draw_image("image", image, scale)

        if cv2.getWindowProperty('image', 1) < 0:
            return False
        return self.wait_key()
