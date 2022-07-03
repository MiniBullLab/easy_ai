#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.loss.utility.base_loss import *


class BasePointNetLoss(BaseLoss):

    def __init__(self, name):
        super().__init__(name)

    def feature_transform_reguliarzer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
        return loss
