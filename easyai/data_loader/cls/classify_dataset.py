#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.cls.classify_dataset_process import ClassifyDatasetProcess
from easyai.data_loader.cls.classify_data_augment import ClassifyDataAugment
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.ClassifyDataSet)
class ClassifyDataSet(TorchDataLoader):

    def __init__(self, data_path, resize_type, normalize_type,
                 mean=0, std=1, image_size=(416, 416),
                 data_channel=3, is_augment=False, transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.image_size = tuple(image_size)
        self.is_augment = is_augment
        self.classify_sample = ClassifySample(data_path)
        self.classify_sample.read_sample()
        self.dataset_process = ClassifyDatasetProcess(resize_type, normalize_type,
                                                      mean, std,
                                                      pad_color=self.get_pad_color())
        self.dataset_augment = ClassifyDataAugment(image_size)

    def __getitem__(self, index):
        img_path, label = self.classify_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        if self.is_augment:
            image = self.dataset_augment.augment(image)
        image = self.dataset_process.normalize_image(image)
        return {'image': image, 'label': label}

    def __len__(self):
        return self.classify_sample.get_sample_count()
