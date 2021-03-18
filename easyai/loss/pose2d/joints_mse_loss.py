#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.registry import REGISTERED_POSE2D_LOSS


@REGISTERED_POSE2D_LOSS.register_module(LossName.JointsMSELoss)
class JointsMSELoss(BaseLoss):

    def __init__(self, reduction, input_size, points_count):
        super().__init__(LossName.JointsMSELoss)
        self.reduction = reduction
        self.input_size = input_size
        self.points_count = points_count
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = True
        self.sigma = 2
        self.heatmap_size = input_size / reduction

    def build_gaussian_map(self, targets):
        result_target = []
        result_weight = []
        for target in targets:
            heatmap = np.zeros((self.points_count,
                                self.heatmap_size[1],
                                self.heatmap_size[0]),
                               dtype=np.float32)
            target_weight = np.ones((self.points_count, 1), dtype=np.float32)
            tmp_size = self.sigma * 3
            for joint_id in range(self.points_count):
                if target[joint_id][0] < 0 or target[joint_id][1] < 0:
                    target_weight[joint_id] = 0
                    continue
                mu_x = int(target[joint_id][0] / self.reduction + 0.5)
                mu_y = int(target[joint_id][1] / self.reduction + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue
                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                heatmap[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
            heatmap = torch.from_numpy(heatmap)
            target_weight = torch.from_numpy(target_weight)
            result_target.append(heatmap)
            result_weight.append(target_weight)
        return torch.cat(result_target, dim=0), torch.cat(result_weight, dim=0)

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (Tensor))
            targets (Tensor)

        Returns:
            loss (Tensor)
        """
        if targets is None:
            return outputs
        else:
            device = outputs.device
            batch_size = outputs.size(0)
            heatmaps_gt, target_weight = self.build_gaussian_map(targets.detach())
            heatmaps_gt = heatmaps_gt.to(device)
            target_weight = heatmaps_gt.to(device)
            heatmaps_pred = outputs.reshape((batch_size, self.points_count, -1)).split(1, 1)
            heatmaps_gt = targets.reshape((batch_size, self.points_count, -1)).split(1, 1)
            loss = 0
            for idx in range(self.points_count):
                heatmap_pred = heatmaps_pred[idx].squeeze()
                heatmap_gt = heatmaps_gt[idx].squeeze()
                if self.use_target_weight:
                    loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]),
                                                 heatmap_gt.mul(target_weight[:, idx]))
                else:
                    loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

            return loss / self.points_count

