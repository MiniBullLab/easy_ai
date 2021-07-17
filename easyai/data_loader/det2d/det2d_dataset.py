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
        self.image_size = tuple(image_size)
        self.detection_sample = DetectionSample(data_path,
                                                detect2d_class,
                                                False)
        self.detection_sample.read_sample()
        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        boxes = self.detection_sample.get_sample_boxes(label_path)
        image, labels = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            boxes,
                                                            self.detect2d_class)
        image = self.dataset_process.normalize_image(image)
        labels = self.dataset_process.normalize_labels(labels, self.image_size)
        labels = self.dataset_process.change_outside_labels(labels)
        torch_labels = self.dataset_process.numpy_to_torch(labels, flag=0)
        src_size = np.array([src_image.shape[1], src_image.shape[0]])  # [width, height]
        src_size = self.dataset_process.numpy_to_torch(src_size, flag=0)
        return {'image': image, 'label': torch_labels,
                'image_path': img_path, "src_size": src_size}

    def __len__(self):
        return self.detection_sample.get_sample_count()
