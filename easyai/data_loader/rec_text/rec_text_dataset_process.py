#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess
from easyai.data_loader.common.rec_text_process import RecTextProcess


class RecTextDataSetProcess(Polygon2dDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.text_process = RecTextProcess()

    def read_character(self, char_path):
        return self.text_process.read_character(char_path)

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

    def width_pad_images(self, img, target_width, pad_type=1):
        """
        将图像进行高度不变，宽度的调整的pad
        :param img:    待pad的图像
        :param target_width:   目标宽度 (image width <= target_width)
        :param pad_type: (1, 2)
        :return:    pad完成后的图像
        """
        padding_img = None
        if pad_type == 1:
            channels, height, width = img.shape
            padding_img = np.zeros((channels, height, target_width), dtype=img.dtype)
            padding_img[:, :, 0:width] = img
            # if target_width != width:  # add border Pad
            #     padding_im[:, :, padding_im:] = img[:, :, width - 1].\
            #         unsqueeze(2).expand(channels, height, target_width - width)
        elif pad_type == 2:
            channels, height, width = img.shape
            if width < target_width:
                left_width = (target_width - width) // 2
                padding_img_left = np.zeros((channels, height, left_width), dtype=img.dtype)
                padding_img = np.concatenate([padding_img_left, img], axis=2)
                right_width = target_width - padding_img.shape[2]
                padding_img_right = np.zeros((channels, height, right_width), dtype=img.dtype)
                padding_img = np.concatenate([padding_img, padding_img_right], axis=2)
            else:
                padding_img = img
        return padding_img



