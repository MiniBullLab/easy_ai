#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import cv2


class ImageShow():

    def __init__(self):
        pass

    def show(self, image, scale=1.0):
        cv2.namedWindow("image", 0)
        cv2.resizeWindow("image", int(image.shape[1] * scale), int(image.shape[0] * scale))
        cv2.imshow('image', image)
        if cv2.waitKey() & 0xff == ord('q'):  # 按q退出
            return False
        else:
            return True
