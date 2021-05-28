#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.keypoint2d import dsntnn
from easyai.loss.utility.registry import REGISTERED_KEYPOINT2D_LOSS


@REGISTERED_KEYPOINT2D_LOSS.register_module(LossName.DSNTLoss)
class DSNTLoss(BaseLoss):

    def __init__(self, input_size, points_count):
        super().__init__(LossName.DSNTLoss)
        self.input_size = input_size
        self.points_count = points_count

    def build_targets(self, targets):
        result = targets.detach()
        for target in result:
            for index in range(self.points_count):
                target[index][0] = (target[index][0] * 2 + 1) / self.image_size[0] - 1  # [-1,1]
                target[index][1] = (target[index][1] * 2 + 1) / self.image_size[1] - 1  # [-1,1]
        return result

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (Tensor))
            targets (Tensor)

        Returns:
            loss (Tensor)
        """
        # Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(outputs)
        # Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        if targets is None:
            return coords
        else:
            result_targets = self.build_targets(targets)
            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, result_targets)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, result_targets, sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)
            return loss
