#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.SegmentDataSetCollate)
class SegmentDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        labels = []
        images = []
        for all_data in batch_list:
            images.append(all_data['image'])
            labels.append(all_data['label'])
        images = torch.stack(images)
        labels = torch.stack(labels)
        return {'image': images, 'label': labels}
