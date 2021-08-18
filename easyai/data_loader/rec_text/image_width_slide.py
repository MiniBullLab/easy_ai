#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.rec_text.rec_text_dataset_process import RecTextDataSetProcess
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess
from easyai.name_manager.dataloader_name import DataTansformsName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATA_TRANSFORMS


@REGISTERED_DATA_TRANSFORMS.register_module(DataTansformsName.ImageWidthSlide)
class ImageWidthSlide():

    def __init__(self, image_size, windows=(24, 32, 40), step=4):
        self.image_size = image_size
        self.windows = windows
        self.step = step
        self.image_process = ImageDataSetProcess()
        self.dataset_process = RecTextDataSetProcess(0, 0)

    def __call__(self, image):
        image = self.dataset_process.width_pad_images(image, self.image_size[0], 2)
        image = self.slide_image(image)
        return image

    def slide_image(self, image):
        _, h, w = image.shape
        output_image = []
        half_of_max_window = max(self.windows) // 2  # 从最大窗口的中线开始滑动，每次移动step的距离
        for center_axis in range(half_of_max_window, w - half_of_max_window, self.step):
            slice_channel = []
            for window_size in self.windows:
                # print("window_size:", window_size)
                # print(center_axis - window_size // 2, center_axis + window_size // 2)
                image_slice = image[:, :, center_axis - window_size // 2: center_axis + window_size // 2]
                image_slice = image_slice.transpose(1, 2, 0)
                image_slice = self.image_process.resize(image_slice, (h, h), 1)
                image_slice = image_slice.transpose(2, 0, 1)
                slice_channel.append(image_slice)
            output_image.append(np.concatenate(slice_channel, axis=0))
        return np.asarray(output_image, dtype=np.float32)