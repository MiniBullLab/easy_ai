#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.pose2d import dsntnn
from easyai.loss.utility.registry import REGISTERED_POSE_LOSS


class DSNTLoss(BaseLoss):

    def __init__(self):
        super().__init__(LossName.DSNTLoss)

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
            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, targets)
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, targets, sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)
            return loss
