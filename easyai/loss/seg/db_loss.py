#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_SEG_LOSS


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6, return_origin=False):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.return_origin = return_origin

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if self.return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


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
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, input_data, batch_data=None):
        input_data = input_data.float()
        if batch_data is not None:
            device = input_data.device
            shrink_maps = input_data[:, 0, :, :]
            threshold_maps = input_data[:, 1, :, :]
            binary_maps = input_data[:, 2, :, :]

            loss_shrink_maps = self.bce_loss(shrink_maps,
                                             batch_data['shrink_map'].to(device),
                                             batch_data['shrink_mask'].to(device))
            loss_threshold_maps = self.l1_loss(threshold_maps,
                                               batch_data['threshold_map'].to(device),
                                               batch_data['threshold_mask'].to(device))
            self.loss_info = dict(loss_shrink_maps=loss_shrink_maps,
                                  loss_threshold_maps=loss_threshold_maps)
            if input_data.size()[1] > 2:
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
