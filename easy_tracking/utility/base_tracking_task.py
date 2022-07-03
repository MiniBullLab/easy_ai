#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
from easyai.helper import DirProcess
from easyai.helper import VideoProcess
from easyai.utility.registry import build_from_cfg
from easy_tracking.utility.tracking_registry import REGISTERED_REID
from easy_tracking.utility.tracking_registry import REGISTERED_TRACKER


class BaseTrackingTask():

    def __init__(self, config_path):
        self.config_path = config_path
        self.dir_process = DirProcess()
        self.video_process = VideoProcess()
        self.reid_task = None
        self.multi_tracker = None

    def build_reid_task(self, reid_task_name, gpu_id):
        reid_task_config = {"type": reid_task_name.strip(),
                            "model_name": None,
                            "gpu_id": gpu_id,
                            "config_path": self.config_path}
        try:
            if REGISTERED_REID.has_class(reid_task_name.strip()):
                self.reid_task = build_from_cfg(reid_task_config, REGISTERED_REID)
            else:
                EasyLogger.error("%s reid task not exits" % reid_task_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)

    def build_tracker_task(self, tracker_name):
        tracker_task_config = {"type": tracker_name.strip()}
        try:
            if REGISTERED_TRACKER.has_class(tracker_name.strip()):
                self.multi_tracker = build_from_cfg(tracker_task_config, REGISTERED_TRACKER)
            else:
                EasyLogger.error("%s tracker task not exits" % tracker_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)


