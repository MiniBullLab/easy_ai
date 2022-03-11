#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easy_tracking.utility.tracking_task_name import TaskName
from easyai.config.utility.image_task_config import ImageTaskConfig
from easyai.config.utility.config_registry import REGISTERED_TASK_CONFIG


@REGISTERED_TASK_CONFIG.register_module(TaskName.Det2d_Mot_Task)
class DetPose2dConfig(ImageTaskConfig):

    def __init__(self):
        super().__init__(TaskName.Det2d_Mot_Task)

        self.config_path = os.path.join(self.config_save_dir, "det2d_mot.json")
