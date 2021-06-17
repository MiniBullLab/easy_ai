#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.helper.json_process import JsonProcess
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.keypoint2d.keypoint2d_dataset_process import KeyPoint2dDataSetProcess
from easyai.data_loader.keypoint2d.batch_dataset_merge import detection_data_merge
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.KeyPoint2dDataset)
class KeyPoint2dDataset(TorchDataLoader):

    def __init__(self, data_path, class_name,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 points_count=9):
        super().__init__(data_path, data_channel)
        self.class_name = class_name
        self.image_size = image_size
        self.detection_sample = DetectionSample(data_path,
                                                class_name,
                                                False)
        self.detection_sample.read_sample()

        self.json_process = JsonProcess()
        self.dataset_process = KeyPoint2dDataSetProcess(points_count, resize_type, normalize_type,
                                                        mean, std, self.get_pad_color())

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        _, keypoints = self.json_process.parse_key_points_data(label_path)
        image, labels = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            keypoints,
                                                            self.class_name)
        torch_image = self.dataset_process.normalize_image(image)

        labels = self.dataset_process.normalize_labels(labels, self.image_size)
        labels = self.dataset_process.change_outside_labels(labels)

        torch_label = self.dataset_process.numpy_to_torch(labels, flag=0)
        return torch_image, torch_label

    def __len__(self):
        return self.detection_sample.get_sample_count()
