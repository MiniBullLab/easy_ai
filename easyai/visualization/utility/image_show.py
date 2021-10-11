#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.visualization.utility.base_show import BaseShow


class ImageShow(BaseShow):

    def __init__(self):
        super().__init__()

    def show(self, image, scale=1.0):
        self.drawing.draw_image('image', image, scale)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
