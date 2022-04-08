#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET

from easy_pc.dataloader.pc_cls.classify_pc_sample import ClassifyPointCloudSample
from easy_pc.dataloader.pc_cls.classify_pc_augment import ClassifyPointCloudAugment
from easy_pc.dataloader.pc_cls.classify_pc_dataset_process import ClassifyPointCloudDatasetProcess
from easy_pc.name_manager.pc_dataloader_name import PCDatasetName


@REGISTERED_DATASET.register_module(PCDatasetName.ClassifyPointCloudDataSet)
class ClassifyPointCloudDataSet(TorchDataLoader):

    def __init__(self, data_path, point_features, is_augment=False,
                 transform_func=None):
        super().__init__(data_path, point_features, transform_func)
        self.is_augment = is_augment
        self.classify_sample = ClassifyPointCloudSample(data_path)
        self.classify_sample.read_sample()

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
