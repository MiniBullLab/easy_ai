#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.task_show.base_show import BaseShow


class SegmentionShow(BaseShow):

    def __init__(self):
        super().__init__()

    def show(self, src_image, result, class_name, scale=0.5):
        segment_image = self.drawing.draw_segment_result(src_image, result,
                                                         class_name)

        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(segment_image.shape[1] * scale), int(segment_image.shape[0] * scale))
        cv2.imshow('image', segment_image)
        if cv2.getWindowProperty('image', 1) < 0:
            return False
        return self.wait_key()
