#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import numpy as np
from easyai.utility.logger import EasyLogger

from easy_pc.dataloader.utility import box3d_op
from easy_pc.dataloader.utility.base_pc_detection_sample import BasePCDetectionSample
from easy_pc.dataloader.pc_det3d.det3d_pc_dataset_sampling import Det3dPCDatasetSampling


class Det3dPointCloudSample(BasePCDetectionSample):

    def __init__(self, train_path, class_name,
                 sample_groups, point_features):
        super().__init__()
        self.train_path = train_path
        self.class_name = class_name
        path, _ = os.path.split(train_path)
        self.data_root = os.path.join(path, "../")
        self.info_path = os.path.join(self.data_root, "dbinfos_train.pkl")
        self.db_sampler = Det3dPCDatasetSampling(self.data_root,
                                                 self.info_path,
                                                 point_features,
                                                 sample_groups,
                                                 class_name)

        self.pc_and_label_list = []
        self.sample_count = 0

    def read_sample(self):
        try:
            self.pc_and_label_list = self.get_pc_and_label_list(self.train_path)
            self.sample_count = self.get_sample_count()
            EasyLogger.warn("%s sample count: %d" % (self.train_path,
                                                     self.sample_count))
        except ValueError as err:
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(err)

    def get_sample_boxes(self, label_path):
        result = []
        _, boxes = self.json_process.parse_rect3d_data(label_path)
        for box in boxes:
            if box.name in self.class_name:
                result.append(box)
        return result

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        pc_path, label_path = self.pc_and_label_list[temp_index]
        return pc_path, label_path

    def get_sample_count(self):
        result = len(self.pc_and_label_list)
        return result

    def object_sample(self, gt_bboxes_3d, gt_labels_3d, points):
        sampled_dict = self.db_sampler.sample_all(gt_bboxes_3d, gt_labels_3d)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate((gt_labels_3d,
                                           sampled_gt_labels),
                                          axis=0)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d,
                                           sampled_gt_bboxes_3d])

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = np.concatenate((sampled_points, points), axis=0)

        return points, gt_bboxes_3d, gt_labels_3d.astype(np.long)

    def remove_points_in_boxes(self, points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box3d_op.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

