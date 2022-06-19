#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import torch.nn.functional as F
from easyai.data_loader.utility.dataloader_registry import REGISTERED_BATCH_DATA_PROCESS

from easy_pc.ops.voxel.voxelize import voxelization
from easy_pc.name_manager.pc_dataloader_name import PCBatchDataProcessName


@REGISTERED_BATCH_DATA_PROCESS.register_module(PCBatchDataProcessName.VoxelizationProcess)
class VoxelizationProcess():

    def __init__(self, voxel_size, point_cloud_range,
                 max_num_points, max_voxels=20000,
                 deterministic=True):
        super().__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def __call__(self, batch_list, device):
        points = batch_list['point_cloud']
        with torch.no_grad():
            voxels, coors, num_points = [], [], []
            for res in points:
                res_voxels, res_coors, res_num_points = voxelization(res,
                                                                     self.voxel_size,
                                                                     self.point_cloud_range,
                                                                     self.max_num_points,
                                                                     self.max_voxels,
                                                                     self.deterministic)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
            voxels = torch.cat(voxels, dim=0)
            num_points = torch.cat(num_points, dim=0)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)
            return voxels.to(self.device), \
                   num_points.to(self.device), \
                   coors_batch.to(self.device)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
