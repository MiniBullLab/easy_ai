#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.pose2d.pose2d_sample import Pose2dSample
from easyai.data_loader.landmark.landmark_dataset_process import LandmarkDataSetProcess
from easyai.data_loader.landmark.landmark_augment import LandmarkDataAugment
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.LandmarkDataset)
class LandmarkDataset(TorchDataLoader):

    def __init__(self, data_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 points_count=68, is_augment=False):
        super().__init__(data_path, data_channel)
        self.class_name = class_name
        self.image_size = image_size
        self.is_augment = is_augment
        self.expand_ratio = 0.1
        self.pose2d_sample = Pose2dSample(data_path, class_name)
        self.pose2d_sample.read_sample()

        self.dataset_process = LandmarkDataSetProcess(resize_type, normalize_type,
                                                      mean, std, self.get_pad_color())

        self.dataset_augment = LandmarkDataAugment()

    def __getitem__(self, index):
        img_path, box = self.pose2d_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image, expand_box = self.dataset_process.crop_image(src_image, box, self.expand_ratio, self.is_augment)
        keypoint = self.dataset_process.crop_label(box, expand_box)
        image, label = self.dataset_process.resize_dataset(image,
                                                           self.image_size,
                                                           keypoint,
                                                           self.class_name)

        if self.is_augment:
            image, label = self.dataset_augment.augment(image, label)

        torch_image = self.dataset_process.normalize_image(image)

        points, box = self.dataset_process.normalize_label(label)
        torch_points = self.dataset_process.numpy_to_torch(points, flag=0)
        torch_box = self.dataset_process.numpy_to_torch(box, flag=0)

        return torch_image, torch_points, torch_box

    def __len__(self):
        return self.pose2d_sample.get_sample_count()
