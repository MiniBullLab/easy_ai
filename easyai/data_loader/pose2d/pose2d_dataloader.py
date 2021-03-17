#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.pose2d.pose2d_sample import Pose2dSample
from easyai.data_loader.pose2d.pose2d_dataset_process import Pose2dDataSetProcess


class Pose2dDataLoader(TorchDataLoader):

    def __init__(self, data_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 points_count=9):
        super().__init__(data_channel)
        self.class_name = class_name
        self.image_size = image_size
        self.expand_ratio = 0.15
        self.pose2d_sample = Pose2dSample(data_path, class_name)
        self.pose2d_sample.read_sample()

        self.dataset_process = Pose2dDataSetProcess(resize_type, normalize_type,
                                                    mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, box = self.pose2d_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image, box = self.dataset_process.expand_dataset(src_image, box, self.expand_ratio)
        image, box = self.dataset_process.resize_dataset(image,
                                                         self.image_size,
                                                         box,
                                                         self.class_name)
        torch_image = self.dataset_process.normalize_image(image)

        label = self.dataset_process.normalize_label(box, self.image_size)
        torch_label = self.dataset_process.numpy_to_torch(label, flag=0)

        return torch_image, torch_label

    def __len__(self):
        return self.pose2d_sample.get_sample_count()


def get_pose2d_train_dataloader(train_path, class_name, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    points_count = data_config.points_count
    batch_size = data_config.train_batch_size
    dataloader = Pose2dDataLoader(train_path, class_name,
                                  resize_type, normalize_type, mean, std,
                                  image_size, data_channel,
                                  points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_poes2d_val_dataloader(val_path, class_name, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    points_count = data_config.points_count
    batch_size = 1
    dataloader = Pose2dDataLoader(val_path, class_name,
                                  resize_type, normalize_type, mean, std,
                                  image_size, data_channel,
                                  points_count)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result




