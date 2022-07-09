#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE

from easy_pc.name_manager.pc_dataloader_name import PCDatasetCollateName


@REGISTERED_DATASET_COLLATE.register_module(PCDatasetCollateName.Det3dPointCloudDataSetCollate)
class Det3dPointCloudDataSetCollate(BaseDatasetCollate):

    def __init__(self):
        super().__init__()

    def __call__(self, batch_list):
        labels = []
        pc_list = []
        box3d_list = []
        label3d_list = []
        pc_path_list = []
        for i in range(len(batch_list)):
            pc_list.append(batch_list[i]['point_cloud'])
            box3d_list.append(batch_list[i]['gt_bboxes_3d'])
            label3d_list.append(batch_list[i]['gt_labels_3d'])
            pc_path_list.append(batch_list[i]['pc_path'])
        return {'point_cloud': pc_list,
                'gt_bboxes_3d': box3d_list,
                'gt_labels_3d': label3d_list,
                'pc_path': pc_path_list}
