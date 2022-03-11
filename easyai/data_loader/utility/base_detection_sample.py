#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
from easyai.data_loader.utility.base_sample import BaseSample
from easyai.utility.logger import EasyLogger


class BaseDetectionSample(BaseSample):

    def __init__(self):
        super().__init__()
        self.annotation_post = ".json"

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
                EasyLogger.error("%s or %s not exist" % (annotation_path, image_path))
        return result

    def get_tracking_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        root_path = os.path.join(path, "../")
        for video_frame_path in self.dirProcess.getFileData(train_path):
            temp_path, filename_and_post = os.path.split(video_frame_path)
            filename, post = os.path.splitext(filename_and_post)
            annotation_filename = filename + self.annotation_post
            annotation_dir = os.path.join(root_path, temp_path, "../Annotations")
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            image_path = os.path.join(root_path, video_frame_path)
            # print(image_path, annotation_path)
            if os.path.exists(annotation_path) and \
                    os.path.exists(image_path):
                result.append((image_path, annotation_path))
            else:
                EasyLogger.error("%s or %s not exist" % (annotation_path, image_path))
        return result
