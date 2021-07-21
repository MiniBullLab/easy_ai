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

    def __init__(self, is_padding=True, pad_value=0):
        super().__init__()
        self.is_padding = is_padding
        self.pad_value = pad_value

    def __call__(self, batch_list):
        max_img_w = max([data['image'].shape[-1] for data in batch_list])
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        length = [len(data['text']) for data in batch_list]
        resize_images = []
        text_list = []
        targets = []
        batch_max_length = max(length)
        for all_data in batch_list:
            if self.is_padding:
                img = self.width_pad_img(all_data['image'], max_img_w)
                resize_images.append(torch.tensor(img, dtype=torch.float))
            else:
                resize_images.append(torch.tensor(all_data['image'], dtype=torch.float))
            text_list.append(all_data['text'])
            text_code = all_data['targets']
            text_code.extend([0] * (batch_max_length - len(all_data['text'])))
            targets.append(text_code)
        targets = torch.tensor(targets, dtype=torch.long)
        targets_lengths = torch.tensor(length, dtype=torch.long)
        resize_images = torch.stack(resize_images)
        # print(resize_images.shape)
        result_data = {'image': resize_images,
                       'label': text_list,
                       'targets': targets,
                       'targets_lengths': targets_lengths}
        return result_data

    def width_pad_img(self, img, target_width):
        """
        将图像进行高度不变，宽度的调整的pad
        :param _img:    待pad的图像
        :param _target_width:   目标宽度
        :return:    pad完成后的图像
        """
        channels, height, width = img.shape
        padding_im = np.zeros((channels, height, target_width), dtype=img.dtype)
        padding_im[:, :, 0:width] = img
        return padding_im
