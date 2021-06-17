#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.gen_image.gen_image_sample import GenImageSample
from easyai.data_loader.gen_image.gen_image_dataset_process import GenImageDatasetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.GenImageDataset)
class GenImageDataset(TorchDataLoader):

    def __init__(self, data_path, resize_type, normalize_type,
                 mean=0, std=1, image_size=(768, 320), data_channel=3):
        super().__init__(data_path, data_channel)
        self.image_size = tuple(image_size)
        self.gan_sample = GenImageSample(data_path)
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
