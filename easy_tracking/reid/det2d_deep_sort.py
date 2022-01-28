#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.task_name import TaskName
from easyai.helper.data_structure import ReIDObject2d
from easyai.tools.task_tool.bot_inference import BotInference
from easy_tracking.deep_sort.feature_extractor import Extractor


class Det2dDeepSort():

    def __init__(self, model_name, gpu_id, weight_path, config_path):
        self.det2d_task = BotInference(TaskName.Detect2d_Task, 1)
        self.det2d_task.build_task(model_name[0], gpu_id, weight_path[0], config_path)
        self.extractor = Extractor(model_name[1], weight_path[1])

    def process(self, src_image):
        result = []
        detection_objects, _ = self.det2d_task.infer(src_image)
        # filter_objects = []
        # for temp_object in detection_objects:
        #     if temp_object.classIndex in [2, 5, 7]:
        #         filter_objects.append(temp_object)
        reid_list = self.get_reid(detection_objects, src_image)
        for temp_index, temp_reid in enumerate(reid_list):
            temp = ReIDObject2d()
            temp.copy_object2d(detection_objects[temp_index])
            temp.reid = temp_reid[:]
            result.append(temp)
        return result

    def get_reid(self, detection_objects, src_image):
        img_crops = []
        for temp_object in detection_objects:
            x1 = int(temp_object.min_corner.x)
            y1 = int(temp_object.min_corner.y)
            x2 = int(temp_object.max_corner.x)
            y2 = int(temp_object.max_corner.y)
            im = src_image[y1:y2, x1:x2]
            img_crops.append(im[:, :, (2, 1, 0)])
        if len(img_crops) > 0:
            features = self.extractor(img_crops)
        else:
            features = np.array([])
        return features
