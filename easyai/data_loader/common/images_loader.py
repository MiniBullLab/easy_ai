#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.helper import DirProcess
from easyai.data_loader.utility.base_data_loader import *
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class ImagesLoader(DataLoader):

    def __init__(self, input_dir, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(input_dir, data_channel)
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
        temp_files = self.dirProcess.getDirFiles(input_dir, "*.*")
        self.files = list(temp_files)
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
