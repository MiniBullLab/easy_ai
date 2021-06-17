#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.multi_task.multi_task_sample import MultiTaskSample
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.Det2dSegDataset)
class Det2dSegDataset(TorchDataLoader):

    def __init__(self, data_path, detect2d_class, seg_class_name,
                 seg_label_type, resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3):
        super().__init__(data_path, data_channel)
        self.detect2d_class = detect2d_class
        self.seg_number_class = len(seg_class_name)
        self.seg_label_type = seg_label_type
        self.image_size = tuple(image_size)

        self.multi_task_sample = MultiTaskSample(data_path,
                                                 detect2d_class,
                                                 False)
        self.multi_task_sample.read_sample()

        self.det2d_dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                             mean, std, self.get_pad_color())
        self.seg_dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                         mean, std, self.get_pad_color())

        self.seg_label_converter = ConvertSegmentionLable()

    def __getitem__(self, index):
        img_path, label_path, segment_path = self.multi_task_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        segment_label = self.read_seg_label_image(segment_path)

        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.det2d_dataset_process.resize_image(src_size, self.image_size)
        segment_label = self.seg_dataset_process.resize_lable(segment_label, src_size, self.image_size)
        segment_label = self.seg_dataset_process.change_label(segment_label, self.seg_number_class)

        torch_image = self.det2d_dataset_process.normalize_image(image)
        torch_segment = self.seg_dataset_process.numpy_to_torch(segment_label).long()

        return img_path, cv_image, torch_image, torch_segment

    def __len__(self):
        return self.multi_task_sample.get_sample_count()

    def read_seg_label_image(self, label_path):
        if self.seg_label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.seg_label_converter.process_segment_label(label_path,
                                                                  self.seg_label_type,
                                                                  self.seg_number_class)
        return mask
