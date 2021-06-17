#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.Det2dDataset)
class Det2dDataset(TorchDataLoader):

    def __init__(self, data_path, detect2d_class,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3):
        super().__init__(data_path, data_channel)
        self.image_size = image_size
        self.detection_sample = DetectionSample(data_path,
                                                detect2d_class,
                                                False)
        self.detection_sample.read_sample()
        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        cv_image, src_image = self.read_src_image(img_path)
        image = self.dataset_process.resize_image(src_image,
                                                  self.image_size)
        image = self.dataset_process.normalize_image(image)
        src_size = np.array([cv_image.shape[1], cv_image.shape[0]])  # [width, height]
        src_size = self.dataset_process.numpy_to_torch(src_size, flag=0)
        return img_path, src_size, image

    def __len__(self):
        return self.detection_sample.get_sample_count()
