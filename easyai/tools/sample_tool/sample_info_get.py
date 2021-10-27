#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.visualization.utility.color_define import SegmentColorDefine
from easyai.config.utility.config_factory import ConfigFactory
from easyai.name_manager.task_name import TaskName
from easyai.utility.logger import EasyLogger


class SampleInformation():

    def __init__(self):
        self.json_process = JsonProcess()
        self.dir_process = DirProcess()
        self.config_factory = ConfigFactory()
        self.images_dir_name = "JPEGImages"
        self.annotation_post = "*.json"

    def create_class_names(self, train_path, task_name, config_path=None):
        result = None
        if task_name.strip() == TaskName.Classify_Task:
            train_task_config = self.config_factory.get_config(task_name, config_path)
            train_task_config.class_name = self.get_classify_class(train_path)
            train_task_config.save_config()
            result = train_task_config.class_name
        elif task_name.strip() == TaskName.Detect2d_Task:
            train_task_config = self.config_factory.get_config(task_name, config_path)
            train_task_config.detect2d_class = self.get_detection_class(train_path)
            train_task_config.save_config()
            result = train_task_config.detect2d_class
        elif task_name.strip() == TaskName.Segment_Task:
            train_task_config = self.config_factory.get_config(task_name, config_path)
            segment_class = self.get_segment_class(train_path)
            all_count = len(SegmentColorDefine.colors)
            if len(SegmentColorDefine.colors) >= len(segment_class):
                segment_class_list = []
                for index, class_name in enumerate(segment_class):
                    color = SegmentColorDefine.colors[index]
                    color_list = [str(i) for i in color]
                    color_str = ','.join(color_list)
                    segment_class_list.append((class_name, color_str))
                color = SegmentColorDefine.background
                color_list = [str(i) for i in color]
                color_str = ','.join(color_list)
                segment_class_list.append(('background', color_str))
                train_task_config.segment_class = segment_class_list
                train_task_config.seg_label_type = 2
                train_task_config.save_config()
                result = train_task_config.segment_class
            else:
                EasyLogger.error("segment class count(%d) > %d" % (len(segment_class),
                                                                   all_count))
                result = None
        assert result is not None
        return result

    def create_NG_OK_class(self):
        train_task_config = self.config_factory.get_config(TaskName.Classify_Task)
        train_task_config.class_name = ('OK', )
        train_task_config.save_config()

    def get_classify_class(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../%s" % self.images_dir_name)
        dir_names = os.listdir(images_dir)
        for name in dir_names:
            if not name.startswith("."):
                file_path = os.path.join(images_dir, name)
                if os.path.isdir(file_path):
                    result.append(name)
        return sorted(result)

    def get_detection_class(self, train_path):
        all_names = []
        path, _ = os.path.split(train_path)
        annotation_dir = os.path.join(path, "../Annotations")
        for label_path in self.dir_process.getDirFiles(annotation_dir, self.annotation_post):
            # print(label_path)
            _, boxes = self.json_process.parse_rect_data(label_path)
            temp_names = [box.name for box in boxes if box.name.strip()]
            all_names.extend(temp_names)
        class_names = list(set(all_names))
        return sorted(class_names)

    def get_segment_class(self, train_path):
        all_names = []
        path, _ = os.path.split(train_path)
        annotation_dir = os.path.join(path, "../Annotations")
        for label_path in self.dir_process.getDirFiles(annotation_dir, self.annotation_post):
            # print(label_path)
            file_name_post, polygon_list = self.json_process.parse_segment_data(label_path)
            temp_names = [polygon.name for polygon in polygon_list if polygon.name.strip()]
            all_names.extend(temp_names)
        class_names = list(set(all_names))
        return sorted(class_names)
