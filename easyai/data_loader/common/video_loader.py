#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper import VideoProcess
from easyai.data_loader.utility.data_loader import *
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.utility.task_dataset_process import TaskDataSetProcess


class VideoLoader(DataLoader):

    def __init__(self, video_path, image_size=(416, 416), data_channel=3,
                 resize_type=0, normalize_type=0, mean=0, std=1):
        super().__init__(data_channel)
        self.video_process = VideoProcess()
        self.image_process = ImageDataSetProcess()
        self.dataset_process = TaskDataSetProcess(resize_type, normalize_type,
                                                  mean, std,
                                                  pad_color=self.get_pad_color())
        if not self.video_process.isVideoFile(video_path) or \
                not self.video_process.openVideo(video_path):
            raise Exception("Invalid path!", video_path)
        self.video_path = video_path
        self.image_size = image_size
        self.normalize_type = normalize_type
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.resize_type = resize_type
        self.count = int(self.video_process.getFrameCount())
        self.image_pad_color = (0, 0, 0)

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        success, cv_image = self.video_process.read_frame()

        if not success:
            raise StopIteration

        src_image = self.read_src_image(cv_image)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        torch_image = self.dataset_process.normalize_image(image)
        torch_image = torch_image.unsqueeze(0)
        video_name = self.video_path + "_%d" % self.index
        return video_name, cv_image, torch_image

    def __len__(self):
        return self.count

    def read_src_image(self, cv_image):
        src_image = None
        if self.data_channel == 1:
            src_image = self.image_process.cv_image_color_convert(cv_image, 0)
        elif self.data_channel == 3:
            src_image = self.image_process.cv_image_color_convert(cv_image, 1)
        else:
            print("read src image error!")
        return src_image
