#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os.path
from easyai.data_loader.utility.base_sample import BaseSample


class BaseClassifySample(BaseSample):

    def __init__(self):
        super().__init__()

    def get_image_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        images_dir = os.path.join(path, "../JPEGImages")
        for line_data in self.dirProcess.getFileData(train_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) == 2:
                image_path = os.path.join(images_dir, data_list[0])
                # print(image_path)
                if os.path.exists(image_path):
                    result.append((image_path, int(data_list[1])))
                else:
                    print("%s not exist" % image_path)
            else:
                print("% error" % line_data)
        return result

    def get_pointcloud_and_label_list(self, train_path):
        result = []
        path, _ = os.path.split(train_path)
        data_dir = os.path.join(path, "../pcds")
        for line_data in self.dirProcess.getFileData(train_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) == 2:
                pcd_path = os.path.join(data_dir, data_list[0])
                # print(pcd_path)
                if os.path.exists(pcd_path):
                    result.append((pcd_path, int(data_list[1])))
                else:
                    print("%s not exist" % pcd_path)
            else:
                print("% error" % line_data)
        return result
