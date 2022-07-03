#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.name_manager.task_name import TaskName
from easyai.helper.data_structure import ReIDObject2d
from easyai.tasks.det2d.detect2d import Detection2d
from easyai.tasks.cls.classify import Classify
from easy_tracking.utility.reid_inference import ReidInference
from easy_tracking.utility.tracking_registry import REGISTERED_REID


@REGISTERED_REID.register_module(TaskName.Det2D_Classify_REID_TASK)
class Det2dDeepSort(ReidInference):

    def __init__(self, model_name, gpu_id, config_path=None):
        super().__init__(model_name, config_path, TaskName.Det2D_Classify_REID_TASK)
        self.det2d_task = Detection2d(None, gpu_id, self.task_config.det_config)
        self.classify_task = Classify(None, gpu_id, self.task_config.classify_config)

    def load_weights(self, weights_path):
        self.det2d_task.load_weights(weights_path[0])
        self.classify_task.load_weights(weights_path[1])

    def process(self, src_image, data_type=1, is_show=False):
        result = []
        input_data = self.det2d_task.get_single_image_data(src_image)
        detection_objects, _ = self.det2d_task.single_image_process(input_data)
        final_objects, reid_list = self.get_reid(detection_objects, src_image)
        for temp_index, temp_object in enumerate(final_objects):
            temp = ReIDObject2d()
            temp.copy_object2d(temp_object)
            temp.reid = reid_list[temp_index][:]
            result.append(temp)
        return result

    def single_image_process(self, input_data):
        pass

    def infer(self, input_data, net_type=0):
        pass

    def get_reid(self, detection_objects, src_image):
        final_objects = []
        img_crops = []
        for temp_object in detection_objects:
            x1 = int(temp_object.min_corner.x)
            y1 = int(temp_object.min_corner.y)
            x2 = int(temp_object.max_corner.x)
            y2 = int(temp_object.max_corner.y)
            im = src_image[y1:y2, x1:x2]
            if im.shape[0] == 0 or im.shape[1] == 0:
                continue
            img_crops.append(im)
            final_objects.append(temp_object)
        if len(img_crops) > 0:
            input_data = self.classify_task.get_single_image_data(img_crops)
            _, temp_reid = self.classify_task.infer(input_data)
            features = temp_reid[0].cpu().numpy()
        else:
            features = np.array([])
        return final_objects, features
