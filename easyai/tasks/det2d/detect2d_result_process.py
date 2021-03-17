#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


import torch
from easyai.helper.dataType import DetectionObject
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.base_algorithm.fast_non_max_suppression import FastNonMaxSuppression
from easyai.tasks.utility.box2d_process import box2d_xywh2xyxy


class Detect2dResultProcess():

    def __init__(self, post_prcoess_type, nms_threshold,
                 image_size, detect2d_class):
        self.post_prcoess_type = post_prcoess_type
        self.nms_threshold = nms_threshold
        self.image_size = image_size
        self.detect2d_class = detect2d_class
        self.min_width = 2  # (pixels)
        self.min_height = 2  # (pixels)
        self.use_new_confidence = False
        self.dataset_process = ImageDataSetProcess()
        self.nms_process = FastNonMaxSuppression()

    def postprocess(self, prediction, src_size, threshold=0.0):
        if prediction is None:
            return None
        result = self.get_detection_result(prediction, threshold,
                                           self.post_prcoess_type)
        detection_objects = self.nms_process.multi_class_nms(result, self.nms_threshold)
        detection_objects = self.resize_detection_objects(src_size,
                                                          self.image_size,
                                                          detection_objects,
                                                          self.detect2d_class)
        return detection_objects

    def get_detection_result(self, prediction, conf_thresh, result_type=0):
        result = None
        if result_type == 0:
            result = self.get_yolo_result(prediction, conf_thresh)
        elif result_type == 1:
            result = self.get_ssd_result(prediction, conf_thresh)
        return result

    def get_yolo_result(self, prediction, conf_thresh):
        result = []
        class_confidence, class_index = prediction[:, 5:].max(1)
        if self.use_new_confidence:
            object_confidence = prediction[:, 4]
            object_confidence *= class_confidence
            temp1_indexs = object_confidence > conf_thresh
        else:
            temp1_indexs = prediction[:, 4] > conf_thresh
        temp2_indexs = (prediction[:, 2:4] > self.min_width).all(1)
        temp3_indexs = torch.isfinite(prediction).all(1)
        index_list = temp1_indexs & temp2_indexs & temp3_indexs
        prediction = prediction[index_list]
        class_confidence = class_confidence[index_list]
        class_index = class_index[index_list]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[:, :4] = box2d_xywh2xyxy(prediction[:, :4])
        for index, value in enumerate(prediction):
            temp_object = DetectionObject()
            temp_object.min_corner.x = value[0]
            temp_object.min_corner.y = value[1]
            temp_object.max_corner.x = value[2]
            temp_object.max_corner.y = value[3]
            temp_object.objectConfidence = value[4]
            temp_object.classConfidence = class_confidence[index]
            temp_object.classIndex = class_index[index]
            result.append(temp_object)
        return result

    def get_ssd_result(self, prediction, conf_thresh):
        result = []
        class_confidence, class_index = prediction[:, 4:].max(1)
        temp1_indexs = class_confidence > conf_thresh
        temp2_indexs = (prediction[:, 2:4] > self.min_width).all(1)
        temp3_indexs = torch.isfinite(prediction).all(1)
        index_list = temp1_indexs & temp2_indexs & temp3_indexs
        prediction = prediction[index_list]
        class_confidence = class_confidence[index_list]
        class_index = class_index[index_list]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[:, :4] = box2d_xywh2xyxy(prediction[:, :4])
        for index, value in enumerate(prediction):
            if class_index[index] != 0:
                temp_object = DetectionObject()
                temp_object.min_corner.x = value[0]
                temp_object.min_corner.y = value[1]
                temp_object.max_corner.x = value[2]
                temp_object.max_corner.y = value[3]
                temp_object.objectConfidence = class_confidence[index]
                temp_object.classConfidence = class_confidence[index]
                temp_object.classIndex = class_index[index] - 1
                result.append(temp_object)
        return result

    def resize_detection_objects(self, src_size, image_size,
                                 detection_objects, class_name):
        result = []
        ratio, pad = self.dataset_process.get_square_size(src_size,
                                                          image_size)
        for obj in detection_objects:
            temp_object = DetectionObject()
            x1 = (obj.min_corner.x - pad[0] // 2) / ratio
            y1 = (obj.min_corner.y - pad[1] // 2) / ratio
            x2 = (obj.max_corner.x - pad[0] // 2) / ratio
            y2 = (obj.max_corner.y - pad[1] // 2) / ratio
            temp_object.min_corner.x = x1
            temp_object.min_corner.y = y1
            temp_object.max_corner.x = x2
            temp_object.max_corner.y = y2
            temp_object.classIndex = int(obj.classIndex)
            temp_object.objectConfidence = obj.objectConfidence
            temp_object.classConfidence = obj.classConfidence
            temp_object.name = class_name[temp_object.classIndex]
            result.append(temp_object)
        return result

