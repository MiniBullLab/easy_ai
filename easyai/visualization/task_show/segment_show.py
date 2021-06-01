#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Segment_Task)
class SegmentionShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Segment_Task)

    def show(self, src_image, result, class_name, scale=0.5):
        image = src_image.copy()
        segment_image = self.drawing.draw_segment_result(image, result,
                                                         class_name)
        self.drawing.draw_image("image", segment_image, scale)

        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
