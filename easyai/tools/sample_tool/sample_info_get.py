#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.config.utility.config_factory import ConfigFactory
from easyai.config.name_manager import TaskName


class SampleInformation():

    def __init__(self):
        self.json_process = JsonProcess()
        self.dir_process = DirProcess()
        self.config_factory = ConfigFactory()
        self.images_dir_name = "JPEGImages"
        self.annotation_post = "*.json"

    def create_class_names(self, train_path, task_name):
        result = None
        if task_name.strip() == TaskName.Classify_Task:
            train_task_config = self.config_factory.get_config(task_name)
            train_task_config.class_name = self.get_classify_class(train_path)
            train_task_config.save_config()
            result = train_task_config.class_name
        elif task_name.strip() == TaskName.Detect2d_Task:
            train_task_config = self.config_factory.get_config(task_name)
            train_task_config.detect2d_class = self.get_detection_class(train_path)
            train_task_config.save_config()
            result = train_task_config.detect2d_class
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
        class_names = set(all_names)
        return tuple(class_names)
