#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.task_show.base_show import BaseShow


class Pose2dShow(BaseShow):

    def __init__(self):
        super().__init__()

    def show(self, src_image, objects_pose, skeleton):
        image = src_image.copy()
        self.drawing.draw_keypoint2d_result(image, objects_pose, skeleton)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(image.shape[1] * 0.8), int(image.shape[0] * 0.8))
        cv2.imshow("image", image)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
