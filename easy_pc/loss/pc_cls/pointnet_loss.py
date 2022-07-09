#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_pc.loss.common.base_pointnet_loss import BasePointNetLoss, F
from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_CLS_LOSS


@REGISTERED_PC_CLS_LOSS.register_module(PCLossName.PointNetClsLoss)
class PointNetClsLoss(BasePointNetLoss):

    def __init__(self, flag):
        super().__init__(PCLossName.PointNetClsLoss)
        self.flag = flag
        self.mat_diff_loss_scale = 0.001

    def forward(self, input_list, batch_data=None):
        if not isinstance(input_list, list):
            input_x = [input_list]
        else:
            input_x = input_list
        x = F.log_softmax(input_x[0], dim=-1)
        if batch_data is not None:
            device = input_x[0].device
            targets = batch_data['label'].to(device)
            loss = F.nll_loss(x, targets.long())
            if self.flag:
                mat_diff_loss = self.feature_transform_reguliarzer(input_x[2])
                loss += mat_diff_loss * self.mat_diff_loss_scale
        else:
            loss = x
        return loss
