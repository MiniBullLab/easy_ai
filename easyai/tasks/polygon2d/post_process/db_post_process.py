#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from easyai.helper.data_structure import Point2d
from easyai.helper.data_structure import Polygon2dObject
from easyai.tasks.utility.base_post_process import BasePostProcess
from easyai.name_manager.post_process_name import PostProcessName
from easyai.tasks.utility.task_registry import REGISTERED_POST_PROCESS


@REGISTERED_POST_PROCESS.register_module(PostProcessName.DBPostProcess)
class DBPostProcess(BasePostProcess):

    def __init__(self, threshold=0.6, mask_threshold=0.3, unclip_ratio=1.5):
        super().__init__()
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.unclip_ratio = unclip_ratio
        self.max_candidates = 1000
        self.min_size = 5
        self.is_output_polygon = False

    def __call__(self, predict_score, src_size):
        result = []
        instance_score = predict_score.squeeze()
        segmentation = instance_score > self.mask_threshold
        bitmap = (segmentation * 255).astype(np.uint8)
        # available_region = np.zeros_like(instance_score, dtype=np.float32)
        # np.putmask(available_region, instance_score > self.mask_threshold, instance_score)
        # mask_region = (available_region > 0).astype(np.uint8) * 255
        # structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # bitmap = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, structure_element)
        # cv2.imwrite("test.png", bitmap)
        if self.is_output_polygon:
            boxes, scores = self.polygons_from_bitmap(instance_score, bitmap, src_size)
        else:
            boxes, scores = self.boxes_from_bitmap(instance_score, bitmap, src_size)
        for box, socre in zip(boxes, scores):
            if socre <= self.threshold:
                continue
            polygon_object = Polygon2dObject()
            polygon_object.clear_polygon()
            for temp_value in box:
                temp_point = Point2d(temp_value[0], temp_value[1])
                polygon_object.add_point(temp_point)
            polygon_object.object_confidence = float(socre)
            result.append(polygon_object)
        return result

    def polygons_from_bitmap(self, pred, bitmap, src_size):
        '''
        bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''

        assert len(bitmap.shape) == 2
        height, width = bitmap.shape
        boxes = []
        scores = []

        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise NotImplementedError(f'opencv {cv2.__version__} not support')

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if score < self.threshold:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * src_size[0]), 0, src_size[0])
            box[:, 1] = np.clip(np.round(box[:, 1] / height * src_size[1]), 0, src_size[1])
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, src_size):
        '''
        bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''

        assert len(bitmap.shape) == 2
        height, width = bitmap.shape
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            raise NotImplementedError(f'opencv {cv2.__version__} not support')
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.threshold > score:
                continue
            box = self.unclip(points, unclip_ratio=self.unclip_ratio)
            if len(box) == 0:
                continue
            box = box.reshape(-1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(np.round(box[:, 0] / width * src_size[0]), 0, src_size[0])
            box[:, 1] = np.clip(np.round(box[:, 1] / height * src_size[1]), 0, src_size[1])
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

