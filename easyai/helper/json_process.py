#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import codecs
import json
from easyai.helper.data_structure import *
from easyai.utility.logger import EasyLogger


class JsonProcess():

    MIN_WIDTH = 10
    MIN_HEIGHT = 10

    def __init__(self):
        pass

    def parse_rect_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect_objects_list = objects_dict['rectObject']
        boxes = []
        for rect_dict in rect_objects_list:
            id = int(rect_dict.get("id", -1))
            class_name = rect_dict['class']
            xmin = rect_dict['minX']
            ymin = rect_dict['minY']
            xmax = rect_dict['maxX']
            ymax = rect_dict['maxY']
            box = Rect2D()
            box.min_corner.x = xmin
            box.min_corner.y = ymin
            box.max_corner.x = xmax
            box.max_corner.y = ymax
            box.name = class_name
            box.objectId = id
            if box.width() >= JsonProcess.MIN_WIDTH \
                    and box.height() >= JsonProcess.MIN_HEIGHT:
                boxes.append(box)
        return image_name, boxes

    def parse_key_points_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect_objects_list = objects_dict['rectObject']
        result = []
        for rect_dict in rect_objects_list:
            class_name = rect_dict['class']
            xmin = rect_dict['minX']
            ymin = rect_dict['minY']
            xmax = rect_dict['maxX']
            ymax = rect_dict['maxY']
            key_points_list = rect_dict.get('keyPoints', None)
            if key_points_list is None:
                continue
            point_count = rect_dict['pointCount']
            keypoint = KeyPoint2D()
            keypoint.min_corner.x = xmin
            keypoint.min_corner.y = ymin
            keypoint.max_corner.x = xmax
            keypoint.max_corner.y = ymax
            keypoint.name = class_name
            keypoint.clear_key_points()
            for index in range(0, point_count, 2):
                point = Point2d(int(key_points_list[index]), int(key_points_list[index+1]))
                keypoint.add_key_points(point)
            result.append(keypoint)
        return image_name, result

    def parse_pose2d_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect_objects_list = objects_dict['rectObject']
        result = []
        for rect_dict in rect_objects_list:
            pose_dict = rect_dict.get('pose', None)
            if pose_dict is None:
                continue

            class_name = rect_dict['class']
            xmin = rect_dict['minX']
            ymin = rect_dict['minY']
            xmax = rect_dict['maxX']
            ymax = rect_dict['maxY']

            pose_index = pose_dict['index']
            skeleton = pose_dict.get('skeleton', ())
            attributes = pose_dict.get('attributes', None)
            flags = ()
            if attributes is not None:
                flags = attributes.get('direction_cls', ())
            else:
                print("attributes not exits")
            keypoint = KeyPoint2D()
            keypoint.min_corner.x = xmin
            keypoint.min_corner.y = ymin
            keypoint.max_corner.x = xmax
            keypoint.max_corner.y = ymax
            keypoint.name = class_name
            keypoint.clear_key_points()
            is_available = False
            for index_name in pose_index:
                temp_data = pose_dict.get(index_name, (-1, 1, 0))
                if isinstance(temp_data[0], (list, tuple)):
                    for point_data in temp_data:
                        flag = int(point_data[2])
                        if flag > 0:
                            is_available = True
                            point = Point2d(int(temp_data[0]), int(temp_data[1]))
                        else:
                            point = Point2d(-1, -1)
                        keypoint.add_key_points(point)
                else:
                    flag = int(temp_data[2])
                    if flag > 0:
                        is_available = True
                        point = Point2d(int(temp_data[0]), int(temp_data[1]))
                    else:
                        point = Point2d(-1, -1)
                    keypoint.add_key_points(point)
            keypoint.set_key_points_flag(flags)
            keypoint.set_key_points_skeleton(skeleton)
            if is_available:
                result.append(keypoint)
        return image_name, result

    def parse_ocr_data(self, json_path):
        if not os.path.exists(json_path):
            EasyLogger.error("%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        ocr_objects_list = objects_dict['ocrObject']
        result = []
        for ocr_dict in ocr_objects_list:
            class_name = ocr_dict['class']
            illegibility = int(ocr_dict.get('illegibility', 1))
            transcription = ocr_dict['transcription']
            language = ocr_dict['language']
            points_list = ocr_dict.get('polygon', None)
            point_count = ocr_dict['pointCount']
            if illegibility == 1 or not transcription.strip():
                continue
            if points_list is None or len(points_list) < 8 or point_count < 4:
                EasyLogger.error("{} {}".format(json_path, points_list))
                continue
            ocr_object = OCRObject()
            ocr_object.name = class_name
            ocr_object.language = language
            ocr_object.object_text = transcription.strip()
            ocr_object.clear_polygon()
            for index in range(0, point_count * 2, 2):
                point = Point2d(int(points_list[index]), int(points_list[index+1]))
                ocr_object.add_point(point)
            result.append(ocr_object)
        return image_name, result

    def parse_segment_data(self, json_path):
        if not os.path.exists(json_path):
            EasyLogger.error("%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        image_name = data_dict['filename']
        objects_dict = data_dict['objects']
        ocr_objects_list = objects_dict['segmentObject']
        result = []
        for ocr_dict in ocr_objects_list:
            class_name = ocr_dict['class']
            points_list = ocr_dict.get('mask', None)
            point_count = ocr_dict['pointCount']
            if points_list is None or len(points_list) < 6 or point_count < 3:
                EasyLogger.error("{} {}".format(json_path, points_list))
                continue
            ocr_object = Polygon2dObject()
            ocr_object.name = class_name
            ocr_object.clear_polygon()
            for index in range(0, point_count * 2, 2):
                point = Point2d(int(points_list[index]), int(points_list[index+1]))
                ocr_object.add_point(point)
            result.append(ocr_object)
        return image_name, result

    def parse_rect3d_data(self, json_path):
        if not os.path.exists(json_path):
            print("error:%s file not exists" % json_path)
            return None, []
        with codecs.open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        pc_name = data_dict['filename']
        objects_dict = data_dict['objects']
        rect3d_objects_list = objects_dict['rect3DObject']
        boxes = []
        for box_value in rect3d_objects_list:
            id = int(box_value.get("id", -1))
            class_name = box_value['class']
            yaw = -box_value['yaw']  # inverse clockwise
            x = box_value['centerX']
            y = box_value['centerY']
            z = box_value['centerZ']
            width = box_value['width']
            length = box_value['length']
            height = box_value['height']
            box = Rect3D()
            box.objectId = id
            box.center.x = x
            box.center.y = y
            box.center.z = z
            box.size.x = width
            box.size.y = length
            box.size.z = height
            box.rotation.z = yaw
            box.name = class_name
            box.objectId = id
            boxes.append(box)
        return pc_name, boxes
