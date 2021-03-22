#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess


class Pose2dSample():

    def __init__(self, train_path, class_name):
        self.train_path = train_path
        self.class_name = class_name

        self.image_and_box_list = []
        self.sample_count = 0

        self.annotation_post = ".json"
        self.dirProcess = DirProcess()

        self.json_process = JsonProcess()

    def read_sample(self):
        image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.image_and_box_list = self.get_image_and_box_list(image_and_label_list)
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, box = self.image_and_box_list[temp_index]
        return img_path, box

    def get_sample_count(self):
        return len(self.image_and_box_list)

    def get_image_and_box_list(self, image_and_label_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, boxes = self.json_process.parse_pose2d_data(label_path)
            for box in boxes:
                if box.name in self.class_name:
                    result.append((image_path, box))
        return result

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        annotation_dir = os.path.join(path, "../Annotations")
        for filename_and_post in self.dirProcess.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            annotation_filename = filename + self.annotation_post
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            image_path = os.path.join(images_dir, filename_and_post)
            # print(image_path)
            if os.path.exists(annotation_path) and \
                    os.path.exists(image_path):
                result.append((image_path, annotation_path))
            else:
                print("%s or %s not exist" % (annotation_path, image_path))
        return result
