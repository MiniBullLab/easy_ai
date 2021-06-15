#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.sr.super_resolution_sample import SuperResolutionSample
from easyai.data_loader.sr.super_resolution_dataset_process import SuperResolutionDatasetProcess


class SuperResolutionDataloader(TorchDataLoader):

    def __init__(self, data_path, resize_type, normalize_type,
                 mean=0, std=1, image_size=(768, 320),
                 data_channel=3, upscale_factor=3):
        super().__init__(data_path, data_channel)
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.target_size = (image_size[0] * self.upscale_factor,
                            image_size[1] * self.upscale_factor)
        self.sr_sample = SuperResolutionSample(data_path)
        self.sr_sample.read_sample()
        self.dataset_process = SuperResolutionDatasetProcess(resize_type, normalize_type,
                                                             mean, std,
                                                             pad_color=self.get_pad_color())

    def __getitem__(self, index):
        lr_path, hr_path = self.sr_sample.get_sample_path(index)
        _, src_lr_image = self.read_src_image(lr_path)
        _, src_hr_image = self.read_src_image(hr_path)
        image, target = self.dataset_process.resize_dataset(src_lr_image,
                                                            self.image_size,
                                                            src_hr_image,
                                                            self.target_size)
        torch_image, torch_target = self.dataset_process.normalize_dataset(image, target)
        return torch_image, torch_target

    def __len__(self):
        return self.sr_sample.get_sample_count()


def get_sr_train_dataloader(train_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    upscale_factor = data_config.upscale_factor
    dataloader = SuperResolutionDataloader(train_path, resize_type, normalize_type, mean, std,
                                           image_size, data_channel, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_sr_val_dataloader(val_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    upscale_factor = data_config.upscale_factor
    batch_size = data_config.test_batch_size
    dataloader = SuperResolutionDataloader(val_path, resize_type, normalize_type, mean, std,
                                           image_size, data_channel, upscale_factor)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
