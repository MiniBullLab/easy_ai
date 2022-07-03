#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE
from easy_pc.name_manager.pc_dataloader_name import PCDatasetCollateName


@REGISTERED_DATASET_COLLATE.register_module(PCDatasetCollateName.ClassifyPointCloudDataSetCollate)
class ClassifyPointCloudDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        labels = []
        images = []
        for all_data in batch_list:
            images.append(all_data['image'])
            labels.append(all_data['label'])
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return {'point_cloud': images, 'label': labels}
