#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.task_show.base_show import BaseShow


class DetectionShow(BaseShow):

    def __init__(self):
        super().__init__()

    def show(self, src_image, detection_objects):
        self.drawing.drawDetectObjects(src_image, detection_objects)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(src_image.shape[1] * 0.8), int(src_image.shape[0] * 0.8))
        cv2.imshow("image", src_image)
        if cv2.getWindowProperty('image', 1) < 0:
            return False
        return self.wait_key()
