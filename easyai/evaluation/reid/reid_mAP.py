#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
import numpy as np
from easyai.helper.data_structure import DetectionObject
from easyai.helper.dir_process import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.evaluation.utility.calculate_rect_AP import CalculateRectAP
from easyai.evaluation.utility.base_evaluation import BaseEvaluation
from easyai.name_manager.evaluation_name import EvaluationName
from easyai.evaluation.utility.evaluation_registry import REGISTERED_EVALUATION
from easyai.utility.logger import EasyLogger


@REGISTERED_EVALUATION.register_module(EvaluationName.ReidDetectionMeanAp)
class ReidDetectionMeanAp(BaseEvaluation):

    def __init__(self, detect2d_class):
        super().__init__()
        self.class_names = detect2d_class
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.AP_process = CalculateRectAP()

    def reset(self):
        pass

    def eval(self, result_dir, val_path):
        aps = []
        for index, name in enumerate(self.class_names):
            file_path = os.path.join(result_dir, "%s.txt" % name)
            gt_boxes = self.get_gt_boxes(val_path, name)
            detect_boxes = self.get_detect_boxes(file_path)
            recall, precision, ap = self.AP_process.calculate_ap(gt_boxes, detect_boxes, 0.5)
            aps += [ap]
        self.print_evaluation(aps)
        return np.mean(aps), aps

    def result_eval(self, result_path, val_path):
        aps = []
        for index, name in enumerate(self.class_names):
            gt_boxes = self.get_gt_boxes(val_path, name)
            detect_boxes = self.get_detect_boxes(result_path, name)
            recall, precision, ap = self.AP_process.calculate_ap(gt_boxes, detect_boxes, 0.5)
            aps += [ap]
        return np.mean(aps), aps

    def print_evaluation(self, aps):
        EasyLogger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
        EasyLogger.info('Results:')
        for i, ap in enumerate(aps):
            EasyLogger.info("{} : {:.3f}".format(self.class_names[i], ap))
            # print(self.className[i] + '_iou: ' + '{:.3f}'.format(ious[aps.index(ap)]))
        # print('Iou acc: ' + '{:.3f}'.format(np.mean(ious)))

    def get_gt_boxes(self, val_path, class_name):
        result = {}
        detection_samples = DetectionSample(val_path, self.class_names)
        image_annotation_list = detection_samples.get_tracking_image_and_label_list(val_path)
        for image_path, annotation_path in image_annotation_list:
            path, filename_post = os.path.split(image_path)
            temp_path1, image_dir = os.path.split(path)
            temp_path2, video_dir = os.path.split(temp_path1)
            save_str = os.path.join(video_dir, image_dir, filename_post)
            _, boxes = self.json_process.parse_rect_data(annotation_path)
            result_boxes = [box for box in boxes if box.name == class_name]
            result[save_str] = result_boxes
        return result

    def get_detect_boxes(self, result_path, class_name=None):
        result = []
        if not os.path.exists(result_path):
            return result
        if class_name is None:
            for line_data in self.dir_process.getFileData(result_path):
                split_datas = [x.strip() for x in line_data.split(' ') if x.strip()]
                filename_post = split_datas[0].strip()
                # print(filename_post)
                temp_object = DetectionObject()
                temp_object.objectConfidence = float(split_datas[1])
                temp_object.min_corner.x = float(split_datas[2])
                temp_object.min_corner.y = float(split_datas[3])
                temp_object.max_corner.x = float(split_datas[4])
                temp_object.max_corner.y = float(split_datas[5])
                result.append((filename_post, temp_object))
        else:
            for line_data in self.dir_process.getFileData(result_path):
                split_datas = [x.strip() for x in line_data.split('|') if x.strip()]
                filename_post = split_datas[0].strip()
                # print(filename_post)
                for temp_box in split_datas[1:]:
                    box_datas = [x.strip() for x in temp_box.split(' ') if x.strip()]
                    if box_datas[0] != class_name:
                        continue
                    temp_object = DetectionObject()
                    temp_object.objectConfidence = float(box_datas[1])
                    temp_object.min_corner.x = float(box_datas[2])
                    temp_object.min_corner.y = float(box_datas[3])
                    temp_object.max_corner.x = float(box_datas[4])
                    temp_object.max_corner.y = float(box_datas[5])
                    result.append((filename_post, temp_object))
        return result
