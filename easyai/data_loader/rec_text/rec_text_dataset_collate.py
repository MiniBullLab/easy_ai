#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import numpy as np
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.RecTextDataSetCollate)
class RecTextDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()
        self.pad_value = 0

    def __call__(self, batch_list):
        max_img_w = max({data[0].shape[-1] for data in batch_list})
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        length = [len(data[1]['text']) for data in batch_list]
        resize_images = []
        text_list = []
        targets = []
        batch_max_length = max(length)
        for image, label in batch_list:
            img = self.width_pad_img(image, max_img_w)
            resize_images.append(torch.tensor(img, dtype=torch.float))
            text_list.append(label['text'])
            text_code = label['targets']
            text_code.extend([0] * (batch_max_length - len(label['text'])))
            targets.append(text_code)
        targets = torch.tensor(targets, dtype=torch.long)
        targets_lengths = torch.tensor(length, dtype=torch.long)
        resize_images = torch.stack(resize_images)
        # print(resize_images.shape)
        labels = {'text': text_list,
                  'targets': targets,
                  'targets_lengths': targets_lengths}
        return resize_images, labels

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
