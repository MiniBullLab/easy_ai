#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.Det2dDataSetCollate)
class Det2dDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        labels = []
        images = []
        image_path_list = []
        src_size_list = []
        for i in range(len(batch_list)):
            images.append(batch_list[i]['image'])
            labels.append(batch_list[i]['label'])
            image_path_list.append(batch_list[i]['image_path'])
            src_size_list.append(batch_list[i]['src_size'])
        images = torch.stack(images)
        return {'image': images, 'label': labels,
                'image_path': image_path_list, "src_size": src_size_list}
