#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


class SSDPostProcess(BasePostProcess):

    def __init__(self, threshold, nms_threshold):
        super().__init__()
        self.threshold = threshold
        self.nms_threshold = nms_threshold
