#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Det2d_Seg_Task)
class Det2dSegTaskShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Det2d_Seg_Task)

    def show(self, src_image, result, class_name,
             detection_objects, scale=0.8):
        image = src_image.copy()
        segment_image = self.drawing.draw_segment_result(image, result,
                                                         class_name)

        self.drawing.draw_detect_objects(segment_image, detection_objects)

        self.drawing.draw_image("image", segment_image, scale)

        if cv2.getWindowProperty('image', 1) < 0:
            return False
        return self.wait_key()
