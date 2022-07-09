#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_pc.loss.common.base_pointnet_loss import BasePointNetLoss, F
from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_SEG_LOSS


@REGISTERED_PC_SEG_LOSS.register_module(PCLossName.PointNetSegLoss)
class PointNetSegLoss(BasePointNetLoss):

    def __init__(self, flag):
        super().__init__(PCLossName.PointNetSegLoss)
        self.flag = flag
        self.mat_diff_loss_scale = 0.001

    def forward(self, input_list, batch_data=None):
        if not isinstance(input_list, list):
            input_x = [input_list]
        else:
            input_x = input_list
        x = input_x[0]
        batch_size = x.size()[0]
        num_class = x.size()[2]
        n_pts = x.size()[1]
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, num_class), dim=-1)
        x = x.view(batch_size, n_pts, num_class)
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

