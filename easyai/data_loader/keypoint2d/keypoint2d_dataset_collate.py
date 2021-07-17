#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.KeyPoint2dDataSetCollate)
class KeyPoint2dDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        list_labels = []
        list_images = []
        for i in range(len(batch_list)):
            list_images.append(batch_list[i]['image'])
            list_labels.append(batch_list[i]['label'])
        list_images = torch.stack(list_images)
        return {'image': list_images, 'label': list_labels}
