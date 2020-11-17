#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


class GenerateImageResultProcess():

    def __init__(self, input_size):
        self.input_size = input_size

    def get_result_image(self, prediction, result_type=0):
        result = None
        if result_type == 0:
            result = self.gray_resize_image(prediction)
        return result

    def gray_resize_image(self, x):
        # 将x的范围由(-1,1)伸缩到(0,1)
        out = 0.5 * (x + 1)
        out = out.reshape(self.input_size[0], self.input_size[1])
        out *= 255.0
        out = out.clip(0, 255)
        return np.uint8(out)

