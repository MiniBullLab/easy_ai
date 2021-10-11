#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc
import cv2
from easyai.visualization.utility.image_drawing import ImageDrawing


class BaseShow():

    def __init__(self):
        self.task_name = None
        self.drawing = ImageDrawing()

    @abc.abstractmethod
    def show(self, *param, **params):
        pass

    def set_task_name(self, task_name):
        self.task_name = task_name

    def get_task_name(self):
        return self.task_name

    def wait_key(self):
        if cv2.waitKey(0) & 0xff == ord('q'):  # 按q退出
            cv2.destroyAllWindows()
            return False
        else:
            return True
