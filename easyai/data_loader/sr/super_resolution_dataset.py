#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.sr.super_resolution_sample import SuperResolutionSample
from easyai.data_loader.sr.super_resolution_dataset_process import SuperResolutionDatasetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.SuperResolutionDataset)
class SuperResolutionDataset(TorchDataLoader):

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
