#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch.utils.data as data
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.gen_image.gen_image_sample import GenImageSample
from easyai.data_loader.gen_image.gen_image_dataset_process import GenImageDatasetProcess


class GenImageDataloader(TorchDataLoader):

    def __init__(self, train_path, resize_type, normalize_type,
                 mean=0, std=1, image_size=(768, 320), data_channel=3):
        super().__init__(data_channel)
        self.image_size = image_size
        self.gan_sample = GenImageSample(train_path)
        self.gan_sample.read_sample()
        self.dataset_process = GenImageDatasetProcess(resize_type, normalize_type,
                                                      mean, std,
                                                      pad_color=self.get_pad_color())

    def __getitem__(self, index):
        img_path, label = self.gan_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_image(src_image, self.image_size)
        image = self.dataset_process.normalize_image(image)
        return image, label

    def __len__(self):
        return self.gan_sample.get_sample_count()


def get_gen_image_train_dataloader(train_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.train_batch_size
    dataloader = GenImageDataloader(train_path, resize_type, normalize_type, mean, std,
                                    image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=True)
    return result


def get_gen_image_val_dataloader(val_path, data_config, num_workers=8):
    resize_type = data_config.resize_type
    normalize_type = data_config.normalize_type
    mean = data_config.data_mean
    std = data_config.data_std
    image_size = data_config.image_size
    data_channel = data_config.data_channel
    batch_size = data_config.test_batch_size
    dataloader = GenImageDataloader(val_path, resize_type, normalize_type, mean, std,
                                    image_size, data_channel)
    result = data.DataLoader(dataset=dataloader, num_workers=num_workers,
                             batch_size=batch_size, shuffle=False)
    return result
