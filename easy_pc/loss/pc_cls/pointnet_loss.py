#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.loss.utility.base_loss import *

from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_LOSS


class PointNetLoss(BaseLoss):

    def __init__(self, name):
        super().__init__(name)

    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss


@REGISTERED_PC_LOSS.register_module(PCLossName.PointNetClsLoss)
class PointNetClsLoss(PointNetLoss):

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


@REGISTERED_PC_LOSS.register_module(PCLossName.PointNetSegLoss)
class PointNetSegLoss(PointNetLoss):

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
