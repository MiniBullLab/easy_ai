#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.data_loader.utility.base_dataset_collate import BaseDatasetCollate
from easyai.name_manager.dataloader_name import DatasetCollateName
from easyai.data_loader.ocr.make_border_map import MakeBorderMap
from easyai.data_loader.ocr.make_shrink_map import MakeShrinkMap
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE


@REGISTERED_DATASET_COLLATE.register_module(DatasetCollateName.DetOCRDataSetCollate)
class DetOCRDataSetCollate(BaseDatasetCollate):

    def __init__(self, target_type=0):
        super().__init__()
        self.target_type = target_type
        self.border_map = MakeBorderMap()
        self.shrink_map = MakeShrinkMap()

    def __call__(self, batch_list):
        result_data = self.build_images(batch_list)
        target_data = self.build_targets(batch_list)
        result_data.update(target_data)
        return result_data

    def build_images(self, batch_list):
        resize_images = []
        for all_data in batch_list:
            resize_images.append(all_data['image'])
        resize_images = torch.stack(resize_images)
        # print(resize_images.shape)
        result_data = {'image': resize_images}
        return result_data

    def build_targets(self, batch_list):
        target_data = dict()
        if self.target_type == 0:
            threshold_map = []
            threshold_mask = []
            shrink_map = []
            shrink_mask = []
            for all_data in batch_list:
                temp1 = self.border_map(all_data)
                temp2 = self.shrink_map(all_data)
                threshold_map.append(torch.from_numpy(temp1['threshold_map']))
                threshold_mask.append(torch.from_numpy(temp1['threshold_mask']))
                shrink_map.append(torch.from_numpy(temp2['shrink_map']))
                shrink_mask.append(torch.from_numpy(temp2['shrink_mask']))
            threshold_map = torch.stack(threshold_map)
            threshold_mask = torch.stack(threshold_mask)
            shrink_map = torch.stack(shrink_map)
            shrink_mask = torch.stack(shrink_mask)
            target_data = {'threshold_map': threshold_map,
                           'threshold_mask': threshold_mask,
                           'shrink_map': shrink_map,
                           'shrink_mask': shrink_mask}
        elif self.target_type == 1:
            bacth_texts = []
            batch_text_polys = []
            src_size_list = []
            for all_data in batch_list:
                bacth_texts.append(all_data['texts'])
                batch_text_polys.append(all_data['text_polys'])
                src_size_list.append(all_data['src_size'])
            target_data = {'texts': bacth_texts,
                           'text_polys': batch_text_polys,
                           'src_size': src_size_list}
        return target_data

