#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.keypoint2d.pose2d_sample import Pose2dSample
from easyai.data_loader.keypoint2d.pose2d_dataset_process import Pose2dDataSetProcess
from easyai.data_loader.keypoint2d.pose2d_augment import Pose2dDataAugment
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.Pose2dDataset)
class Pose2dDataset(TorchDataLoader):

    def __init__(self, data_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 points_count=17, is_augment=False, transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.class_name = class_name
        self.image_size = tuple(image_size)
        self.is_augment = is_augment
        self.expand_ratio = 0.15
        self.pose2d_sample = Pose2dSample(data_path, class_name)
        self.pose2d_sample.read_sample()

        self.dataset_process = Pose2dDataSetProcess(resize_type, normalize_type,
                                                    mean, std, self.get_pad_color())

        self.dataset_augment = Pose2dDataAugment()

    def __getitem__(self, index):
        img_path, box = self.pose2d_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image, expand_box = self.dataset_process.crop_image(src_image, box, self.expand_ratio)
        keypoint = self.dataset_process.crop_label(box, expand_box)
        image, label = self.dataset_process.resize_dataset(image,
                                                           self.image_size,
                                                           keypoint,
                                                           self.class_name)

        if self.is_augment:
            image, label = self.dataset_augment.augment(image, label)

        torch_image = self.dataset_process.normalize_image(image)

        label = self.dataset_process.normalize_label(label)
        torch_label = self.dataset_process.numpy_to_torch(label, flag=0)

        return {'image': torch_image, 'label': torch_label}

    def __len__(self):
        return self.pose2d_sample.get_sample_count()
