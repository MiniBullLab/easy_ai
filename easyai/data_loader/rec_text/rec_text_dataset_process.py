#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess
from easyai.data_loader.common.rec_text_process import RecTextProcess
from easyai.utility.logger import EasyLogger


class RecTextDataSetProcess(Polygon2dDataSetProcess):

    def __init__(self, char_path, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.text_process = RecTextProcess()
        EasyLogger.debug(char_path)
        self.character = self.text_process.read_character(char_path)

    def normalize_image(self, src_image):
        image = self.dataset_process.normalize(input_data=src_image,
                                               normalize_type=self.normalize_type,
                                               mean=self.mean,
                                               std=self.std)
        image = self.dataset_process.numpy_transpose(image)
        return image

    def normalize_label(self, ocr_object):
        text = ocr_object.get_text()
        text_code, text = self.text_process.text_encode(text)
        result = {'text': text,
                  'targets': text_code}
        return result

    def padding_images(self, image, image_size):
        src_size = (image.shape[1], image.shape[0])  # [width, height]
        if src_size[0] < image_size[0]:
            # Padding
            img = np.concatenate([np.array([[0] * ((image_size[0] - image.shape[1]) // 2)] * 32), image], axis=1)
            img = np.concatenate([img, np.array([[0] * (image_size[0] - img.shape[1])] * 32)], axis=1)
        else:
            img = self.dataset_process.resize(image, image_size, 0)
        return img

    def slide_image(self, image, windows, step):
        _, h, w = image.shape  # No channel for gray image.
        output_image = []
        half_of_max_window = max(windows) // 2  # 从最大窗口的中线开始滑动，每次移动step的距离
        for center_axis in range(half_of_max_window, w - half_of_max_window, step):
            slice_channel = []
            for window_size in windows:
                image_slice = image[:, , center_axis - window_size // 2: center_axis + window_size // 2]
                image_slice = self.dataset_process.resize(image_slice, (32, 32), 0)
                # image_slice = cv2.resize(image_slice, (32, 32))
                slice_channel.append(image_slice)
            output_image.append(np.asarray(slice_channel, dtype=np.float32))
        return np.asarray(output_image, dtype=np.float32)



