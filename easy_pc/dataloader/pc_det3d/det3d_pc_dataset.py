#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger

from easy_pc.helper.pointcloud_process import PointCloudProcess
from easy_pc.name_manager.pc_dataloader_name import PCDatasetName
from easy_pc.dataloader.pc_det3d.det3d_pc_sample import Det3dPointCloudSample
from easy_pc.dataloader.pc_det3d.det3d_pc_augment import Det3dPointCloudAugment
from easy_pc.dataloader.pc_det3d.det3d_pc_dataset_process import Det3dPointCloudDatasetProcess


@REGISTERED_DATASET.register_module(PCDatasetName.Det3dPointCloudDataset)
class Det3dPointCloudDataset(TorchDataLoader):

    def __init__(self, data_path, detect3d_class, point_cloud_range,
                 point_features=4, sample_groups=None,
                 is_augment=False, transform_func=None):
        super().__init__(data_path, point_features, transform_func)
        self.is_augment = is_augment
        self.detect3d_class = detect3d_class
        self.point_cloud_range = point_cloud_range
        self.pointcloud_process = PointCloudProcess(dim=point_features)
        EasyLogger.debug("det3d class: {}".format(detect3d_class))
        self.det3d_sample = Det3dPointCloudSample(data_path,
                                                  detect3d_class,
                                                  sample_groups,
                                                  point_features)
        self.det3d_sample.read_sample()

        self.dataset_process = Det3dPointCloudDatasetProcess(detect3d_class,
                                                             point_cloud_range)

        self.dataset_augment = Det3dPointCloudAugment()

        self.filter_empty_gt = True
        self.flag = np.zeros(self.det3d_sample.get_sample_count(), dtype=np.uint8)

    def __getitem__(self, index):
        pc_path, label_path = self.det3d_sample.get_sample_path(index)
        points = self.pointcloud_process.read_pointcloud(pc_path)
        box3d_list = self.det3d_sample.get_sample_boxes(label_path)
        gt_bboxes_3d, gt_labels_3d = self.dataset_process.convert_labels(box3d_list)
        if self.is_augment:
            points, gt_bboxes_3d, gt_labels_3d = self.det3d_sample.object_sample(gt_bboxes_3d,
                                                                                 gt_labels_3d,
                                                                                 points)
            gt_labels_3d, points = self.dataset_augment.augment(points, gt_bboxes_3d)
        points = self.dataset_process.pc_filtering(points)
        gt_bboxes_3d, gt_labels_3d = self.dataset_process.labels_filtering(gt_bboxes_3d,
                                                                           gt_labels_3d)
        points, gt_bboxes_3d, gt_labels_3d = self.dataset_process.normaliza_dataset(points,
                                                                                    gt_bboxes_3d,
                                                                                    gt_labels_3d)
        return {'point_cloud': points,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels_3d': gt_labels_3d,
                'pc_path': pc_path}

    def __len__(self):
        return self.det3d_sample.get_sample_count()

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
