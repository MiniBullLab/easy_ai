#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import numpy as np
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.rec_text.rec_text_dataset_process import RecTextDataSetProcess
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.RecTextDataSetCollate)
class RecTextDataSetCollate(BaseDatasetCollate):

    def __init__(self, padding_type=0, target_type=0,
                 pad_value=0, character_count=38):
        super().__init__()
        self.dataset_process = RecTextDataSetProcess(0, 0)
        self.padding_type = padding_type
        self.target_type = target_type
        self.pad_value = pad_value
        self.character_count = character_count

    def __call__(self, batch_list):
        result_data = self.build_images(batch_list)
        target_data = self.build_targets(batch_list)
        result_data.update(target_data)
        return result_data

    def build_images(self, batch_list):
        max_img_w = max([data['image'].shape[-1] for data in batch_list])
        max_img_w = int(np.ceil(max_img_w / 8) * 8)
        resize_images = []
        text_list = []
        for all_data in batch_list:
            if self.padding_type > 0:
                img = self.dataset_process.width_pad_images(all_data['image'],
                                                            max_img_w,
                                                            self.padding_type)
                resize_images.append(torch.tensor(img, dtype=torch.float))
            else:
                resize_images.append(torch.tensor(all_data['image'], dtype=torch.float))
            text_list.append(all_data['text'])
        resize_images = torch.stack(resize_images)
        # print(resize_images.shape)
        result_data = {'image': resize_images,
                       'label': text_list}
        return result_data

    def build_targets(self, batch_list):
        target_data = dict()
        if self.target_type == 0:
            length = [len(data['text']) for data in batch_list]
            targets = []
            batch_max_length = max(length)
            for all_data in batch_list:
                text_code = all_data['targets']
                text_code.extend([0] * (batch_max_length - len(all_data['text'])))
                targets.append(text_code)
            targets = torch.tensor(targets, dtype=torch.long)
            targets_lengths = torch.tensor(length, dtype=torch.long)
            target_data = {'targets': targets,
                           'targets_lengths': targets_lengths}
        elif self.target_type == 1:
            targets = []
            for all_data in batch_list:
                label = np.zeros(self.character_count).astype('float32')
                text_code = all_data['targets']
                for ln in text_code:
                    label[int(ln)] += 1  # label construction for ACE
                label[0] = len(text_code)
                targets.append(label)
            targets = torch.tensor(targets)
            target_data = {'targets': targets}
        return target_data


