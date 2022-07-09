#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess

from easy_pc.dataloader.utility import box3d_op


class Det3dPointCloudDatasetProcess(BaseDataSetProcess):

    def __init__(self, detect3d_class, point_cloud_range):
        super().__init__()
        self.detect3d_class = detect3d_class
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def pc_filtering(self, src_cloud):
        in_range_flags = ((src_cloud[:, 0] > self.pcd_range[0])
                          & (src_cloud[:, 1] > self.pcd_range[1])
                          & (src_cloud[:, 2] > self.pcd_range[2])
                          & (src_cloud[:, 0] < self.pcd_range[3])
                          & (src_cloud[:, 1] < self.pcd_range[4])
                          & (src_cloud[:, 2] < self.pcd_range[5]))
        clean_points = src_cloud[in_range_flags]
        return clean_points

    def labels_filtering(self, gt_bboxes_3d, gt_labels_3d):
        bev_range = self.pcd_range[[0, 1, 3, 4]]
        in_range_flags = ((gt_bboxes_3d[:, 0] > bev_range[0])
                          & (gt_bboxes_3d[:, 1] > bev_range[1])
                          & (gt_bboxes_3d[:, 0] < bev_range[2])
                          & (gt_bboxes_3d[:, 1] < bev_range[3]))
        boxes_3d = gt_bboxes_3d[in_range_flags]
        labels_3d = gt_labels_3d[in_range_flags]

        boxes_3d[:, 6] = box3d_op.limit_period(boxes_3d[:, 6],
                                               offset=0.5,
                                               period=2 * np.pi)
        return boxes_3d, labels_3d

    def convert_labels(self, labels):
        box_locs = []
        box_labels = []
        for box3d in labels:
            temp_loc = [box3d.centerX,
                        box3d.centerY,
                        box3d.centerZ,
                        box3d.width,
                        box3d.length,
                        box3d.height,
                        box3d.rotation.z]
            box_locs.append(temp_loc)
            box_labels.append(self.detect3d_class.index(box3d.name))
        gt_bboxes_3d = np.array(box_locs).astype(np.float32)
        gt_labels_3d = np.array(box_labels).astype(np.long)
        return gt_bboxes_3d, gt_labels_3d

    def normaliza_dataset(self, src_cloud, bboxes_3d, labels_3d):
        points = self.numpy_to_torch(src_cloud, flag=0)
        boxes_3d = self.numpy_to_torch(bboxes_3d, flag=0)
        labels_3d = self.numpy_to_torch(labels_3d, flag=0)
        return points, boxes_3d, labels_3d
