#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import numpy as np
from pathlib import Path
from easyai.helper import DirProcess
from easyai.data_loader.utility.base_data_loader import *
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess
from easyai.utility.logger import EasyLogger


class TextDataLoader(DataLoader):

    def __init__(self, input_path, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(input_path, data_channel)
        if Path(input_path).suffix not in ['.txt', '.text']:
            raise Exception("Invalid path!", input_path)
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.image_process = ImageProcess()
        self.dirProcess = DirProcess()
        self.dataset_process = TaskDataSetProcess(resize_type, normalize_type,
                                                  mean, std,
                                                  pad_color=self.get_pad_color())
        self.files = self.get_image_list(input_path)
        self.count = len(self.files)
        self.image_pad_color = (0, 0, 0)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.count:
            raise StopIteration
        image_path = self.files[self.index]
        cv_image, src_image = self.read_src_image(image_path)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        torch_image = torch_image.unsqueeze(0)
        return {"file_path": image_path, "src_image": cv_image,
                "image": torch_image}

    def __len__(self):
        return self.count

    def get_image_list(self, input_path):
        result = []
        path, _ = os.path.split(input_path)
        images_dir = os.path.join(path, "../JPEGImages")
        for line_data in self.dirProcess.getFileData(input_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            if len(data_list) > 0:
                image_path = os.path.join(images_dir, data_list[0])
                # print(image_path)
                if os.path.exists(image_path):
                    result.append(image_path)
                else:
                    EasyLogger.error("%s not exist" % image_path)
            else:
                EasyLogger.warn("% error" % line_data)
        return result
