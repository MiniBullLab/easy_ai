#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

# from easyai.name_manager.loss_name import LossName
# from easyai.loss.common.base_loss import *
# from easyai.loss.common.registry import REGISTERED_COMMON_LOSS
#
#
# @REGISTERED_COMMON_LOSS.register_module(LossName.MeanSquaredErrorLoss)
# class MeanSquaredErrorLoss(BaseLoss):
#
#     def __init__(self, reduction='mean'):
#         super().__init__(LossName.MeanSquaredErrorLoss)
#         self.loss_function = torch.nn.MSELoss(reduction=reduction)
#
#     def forward(self, input_data, target=None):
#         if target is not None:
#             loss = self.loss_function(input_data, target)
#         else:
#             loss = input_data
#         return loss
