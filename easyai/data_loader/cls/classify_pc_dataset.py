#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:eijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.cls.classify_sample import ClassifySample
from easyai.data_loader.cls.classify_pc_augment import ClassifyPointCloudAugment
from easyai.data_loader.cls.classify_pc_dataset_process import ClassifyPointCloudDatasetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET


@REGISTERED_DATASET.register_module(DatasetName.ClassifyPointCloudDataSet)
class ClassifyPointCloudDataSet(TorchDataLoader):

    def __init__(self, data_path, dim, is_augment=False):
        super().__init__(data_path, dim)
        self.is_augment = is_augment
        self.classify_sample = ClassifySample(data_path)
        self.classify_sample.read_sample(flag=1)

        self.pc_augment = ClassifyPointCloudAugment()
        self.pc_process = ClassifyPointCloudDatasetProcess()

    def __getitem__(self, index):
        data_path, label = self.classify_sample.get_sample_path(index)
        point_cloud = self.pointcloud_process.read_pointcloud(data_path)
        if self.is_augment:
            point_cloud = self.pc_augment.augment(point_cloud)
        point_cloud = self.pc_process.normaliza_dataset(point_cloud)
        return {'point_cloud': point_cloud, 'label': label}

    def __len__(self):
        return self.classify_sample.get_sample_count()
