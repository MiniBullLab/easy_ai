#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.data_loader.seg.segment_data_augment import SegmentDataAugment
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger


@REGISTERED_DATASET.register_module(DatasetName.SegmentDataset)
class SegmentDataset(TorchDataLoader):

    def __init__(self, data_path, class_names, label_type,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(768, 320), data_channel=3, is_augment=False):
        super().__init__(data_path, data_channel)
        self.class_names = class_names
        self.label_type = label_type
        self.number_class = len(class_names)
        self.is_augment = is_augment
        self.image_size = tuple(image_size)
        self.segment_sample = SegmentSample(data_path)
        self.segment_sample.read_sample()
        self.dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                     mean, std, self.get_pad_color())
        self.data_augment = SegmentDataAugment()
        self.label_converter = ConvertSegmentionLable()

    def __getitem__(self, index):
        img_path, label_path = self.segment_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        label = self.read_label_image(label_path)
        image, target = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            label)
        if self.is_augment:
            image, target = self.data_augment.augment(image, target)
        target = self.dataset_process.change_label(target, self.number_class)
        torch_image = self.dataset_process.normalize_image(image)
        torch_target = self.dataset_process.numpy_to_torch(target).long()
        return {'image': torch_image, 'label': torch_target}

    def __len__(self):
        return self.segment_sample.get_sample_count()

    def read_label_image(self, label_path):
        if self.label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.label_converter.process_segment_label(label_path,
                                                              self.label_type,
                                                              self.class_names)
        if mask is None:
            EasyLogger.error("segment(%s) label read fail!" % label_path)
        return mask
