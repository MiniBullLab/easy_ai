#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
from easyai.data_loader.utility.base_sample import BaseSample
from easyai.utility.logger import EasyLogger


class BasePCDetectionSample(BaseSample):

    def __init__(self):
        super().__init__()
        self.annotation_post = ".json"

    def get_pc_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        pcd_dir = os.path.join(path, "../pcds")
        annotation_dir = os.path.join(path, "../Annotations")
        for filename_and_post in self.dirProcess.getFileData(train_path):
            filename, post = os.path.splitext(filename_and_post)
            annotation_filename = filename + self.annotation_post
            annotation_path = os.path.join(annotation_dir, annotation_filename)
            pc_path = os.path.join(pcd_dir, filename_and_post)
            # print(pc_path)
            if os.path.exists(annotation_path) and \
                    os.path.exists(pc_path):
                result.append((pc_path, annotation_path))
            else:
                EasyLogger.error("%s or %s not exist" % (annotation_path, pc_path))
        return result
