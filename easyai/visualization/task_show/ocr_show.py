#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
from easyai.name_manager.task_name import TaskName
from easyai.visualization.utility.base_show import BaseShow
from easyai.visualization.utility.show_registry import REGISTERED_TASK_SHOW


@REGISTERED_TASK_SHOW.register_module(TaskName.OCR_Task)
class OCRShow(BaseShow):

    def __init__(self):
        super().__init__()
        self.set_task_name(TaskName.OCR_Task)

    def show(self, src_image, ocr_objects):
        image = self.drawing.draw_ocr_result(src_image, ocr_objects)
        h, w, _, = image.shape
        ratio = 1
        if w > 1200:
            ratio = 1200.0 / w
        image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
        self.drawing.draw_image("image", image)
        if cv2.getWindowProperty('image', 1) < 0:
            return True
        return self.wait_key()
