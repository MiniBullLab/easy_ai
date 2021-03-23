#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.task_show.base_show import BaseShow


class DetAndKeyPoint2dShow(BaseShow):

    def __init__(self):
        super().__init__()

    def show(self, src_image, detection_objects, skeleton):
        self.drawing.draw_det_keypoint2d_result(src_image, detection_objects, skeleton)
        cv2.namedWindow("image", 0)
        # cv2.resizeWindow("image", int(src_image.shape[1] * 0.8), int(src_image.shape[0] * 0.8))
        cv2.imshow("image", src_image)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()