#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.Det_Pose2d_Task)
class DetAndPose2dShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.Det_Pose2d_Task)

    def show(self, src_image, detection_objects, objects_pose, skeleton):
        # image = src_image.copy()
        self.drawing.draw_detect_objects(src_image, detection_objects)
        self.drawing.draw_keypoint2d_result(src_image, objects_pose, skeleton)
        self.drawing.draw_image("image", src_image, 1.0)
        if cv2.getWindowProperty('image', 1) < 0:
            return False
        return self.wait_key()
