#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.seg.seg_common_loss import BalanceCrossEntropyLoss, DiceLoss, MaskL1Loss
from easyai.loss.utility.loss_registry import REGISTERED_SEG_LOSS


@REGISTERED_SEG_LOSS.register_module(LossName.DBLoss)
class DBLoss(BaseLoss):

    def __init__(self, alpha=1.0, beta=10, ohem_ratio=3,
                 reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__(LossName.DBLoss)
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.reduction = reduction

    def forward(self, input_data, batch_data=None):
        input_data = input_data.float()
        if batch_data is not None:
            device = input_data.device
            shrink_maps = input_data[:, 0, :, :]
            loss_shrink_maps = self.bce_loss(shrink_maps,
                                             batch_data['shrink_map'].to(device),
                                             batch_data['shrink_mask'].to(device))

            if input_data.size()[1] > 2:
                threshold_maps = input_data[:, 1, :, :]
                loss_threshold_maps = self.l1_loss(threshold_maps,
                                                   batch_data['threshold_map'].to(device),
                                                   batch_data['threshold_mask'].to(device))
                self.loss_info = dict(loss_shrink_maps=loss_shrink_maps,
                                      loss_threshold_maps=loss_threshold_maps)
                binary_maps = input_data[:, 2, :, :]
                loss_binary_maps = self.dice_loss(binary_maps,
                                                  batch_data['shrink_map'].to(device),
                                                  batch_data['shrink_mask'].to(device))
                self.loss_info['loss_binary_maps'] = loss_binary_maps
                loss = self.alpha * loss_shrink_maps + self.beta * loss_threshold_maps + loss_binary_maps
            else:
                loss = loss_shrink_maps
        else:
            loss = input_data
        return loss
