#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import shutil
from easyai.helper.dir_process import DirProcess
from easyai.utility.logger import EasyLogger


class CopyImage():

    def __init__(self):
        self.images_dir_name = "../JPEGImages"
        self.dir_process = DirProcess()

    def copy(self, train_path, image_save_dir, separator=None):
        image_list = self.get_image_list(train_path, separator)
        if os.path.exists(image_save_dir):
            os.system('rm -rf ' + image_save_dir)
        os.makedirs(image_save_dir, exist_ok=True)
        if len(image_list) > 0:
            image_path = image_list[0]
            path, image_name = os.path.split(image_path)
            save_path = os.path.join(image_save_dir, image_name)
            shutil.copy(image_path, save_path)
        else:
            EasyLogger.error("empty images")

    def get_image_list(self, train_path, separator=None):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, self.images_dir_name)
        for data_line in self.dir_process.getFileData(train_path):
            fileNameAndPost = data_line.split(separator)[0]
            image_path = os.path.join(images_dir, fileNameAndPost)
            # print(image_path)
            if os.path.exists(image_path):
                result.append(image_path)
            else:
                EasyLogger.warn("%s not exists" % image_path)
        return result
