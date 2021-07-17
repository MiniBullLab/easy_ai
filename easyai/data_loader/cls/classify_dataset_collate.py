#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.ClassifyDataSetCollate)
class ClassifyDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        labels = []
        images = []
        for i in range(len(batch_list)):
            labels.append(batch_list[i]['label'])
            images.append(batch_list[i]['image'])
        labels = torch.stack(labels)
        images = torch.stack(images)
        return {'image': images, 'label': labels}
