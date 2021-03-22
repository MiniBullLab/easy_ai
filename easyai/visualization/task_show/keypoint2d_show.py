#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.utility.image_drawing import ImageDrawing


class KeyPoint2dShow():

    def __init__(self):
        self.drawing = ImageDrawing()

    def show(self, src_image, result_objects, skeleton):
        self.drawing.draw_keypoint2d_result(src_image, result_objects, skeleton)
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(src_image.shape[1] * 0.8), int(src_image.shape[0] * 0.8))
        cv2.imshow("image", src_image)
        if cv2.waitKey() & 0xff == ord('q'):
            return False
        else:
            return True
