#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate


class TextDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()
        self.pad_value = 0

    def __call__(self, batch_list):
        max_img_w = max({data[0].shape[-1] for data in batch_list})
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        for image, label in batch_list:
            pass

    def width_pad_img(self, img, target_width):
        """
        将图像进行高度不变，宽度的调整的pad
        :param _img:    待pad的图像
        :param _target_width:   目标宽度
        :return:    pad完成后的图像
        """
        _channels, _height, _width = img.shape
        to_return_img = np.ones([_channels, _height, target_width],
                                dtype=img.dtype) * self.pad_value
        to_return_img[:, :_height, :_width] = img
        return to_return_img
