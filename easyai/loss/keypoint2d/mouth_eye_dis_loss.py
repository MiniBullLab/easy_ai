#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager import LossName
from easyai.loss.utility.base_loss import *


class MouthEyeFrontDisLoss(BaseLoss):
    def __init__(self, ignore_value=-1):
        super().__init__(LossName.MouthEyeFrontDisLoss)
        self.ignore_value = ignore_value

    def forward(self, output, targets):
        device = output.device
        batch_size = output.size(0)
        predictions = output.reshape((batch_size, -1))
        gt_data = targets.reshape((batch_size, -1)).to(device)
        weight_mask = torch.where(gt_data == self.ignore_value,
                                  torch.zeros_like(gt_data),
                                  torch.ones_like(gt_data))
        # left eye
        predictions_left_eye = ((predictions[:, 41*2+1] - predictions[:, 37*2+1]).mul(weight_mask[:, 41*2+1]).mul(weight_mask[:, 37*2+1]) +
                                (predictions[:, 40*2+1] - predictions[:, 38*2+1]).mul(weight_mask[:, 40*2+1]).mul(weight_mask[:, 38*2+1])) / 2

        targets_left_eye = ((gt_data[:, 41*2+1] - gt_data[:, 37*2+1]).mul(weight_mask[:, 41*2+1]).mul(weight_mask[:, 37*2+1]) +
                            (gt_data[:, 40*2+1] - gt_data[:, 38*2+1]).mul(weight_mask[:, 40*2+1]).mul(weight_mask[:, 38*2+1])) / 2
        t_left_eye = torch.abs(predictions_left_eye - targets_left_eye)
        # print('t_left_eye shape:', t_left_eye.size())
        left_eye_loss = torch.where(t_left_eye < 1, 0.5*t_left_eye.mul(t_left_eye), t_left_eye - 0.5)


        # right eye
        predictions_right_eye = ((predictions[:, 47*2+1] - predictions[:, 43*2+1]).mul(weight_mask[:, 47*2+1]).mul(weight_mask[:, 43*2+1]) +
                                 (predictions[:, 46*2+1] - predictions[:, 44*2+1]).mul(weight_mask[:, 46*2+1]).mul(weight_mask[:, 44*2+1])) / 2

        targets_right_eye = ((gt_data[:, 47*2+1] - gt_data[:, 43*2+1]).mul(weight_mask[:, 47*2+1]).mul(weight_mask[:, 43*2+1]) +
                             (gt_data[:, 46*2+1] - gt_data[:, 44*2+1]).mul(weight_mask[:, 46*2+1]).mul(weight_mask[:, 44*2+1])) / 2
        t_right_eye = torch.abs(predictions_right_eye - targets_right_eye)
        # print('t_right_eye shape:', t_right_eye.size())
        right_eye_loss = torch.where(t_right_eye < 1, 0.5*t_right_eye.mul(t_right_eye), t_right_eye - 0.5)


        # mouth
        predictions_mouth = ((predictions[:, 67*2+1] - predictions[:, 61*2+1]).mul(weight_mask[:, 67*2+1]).mul(weight_mask[:, 61*2+1]) +
                             (predictions[:, 65*2+1] - predictions[:, 63*2+1]).mul(weight_mask[:, 65*2+1]).mul(weight_mask[:, 63*2+1])) / 2

        targets_mouth = ((gt_data[:, 67*2+1] - gt_data[:, 61*2+1]).mul(weight_mask[:, 67*2+1]).mul(weight_mask[:, 61*2+1]) +
                         (gt_data[:, 65*2+1] - gt_data[:, 63*2+1]).mul(weight_mask[:, 65*2+1]).mul(weight_mask[:, 63*2+1])) / 2
        t_mouth = torch.abs(predictions_mouth - targets_mouth)
        mouth_loss = torch.where(t_mouth < 1, 0.5*t_mouth.mul(t_mouth), t_mouth - 0.5)

        return torch.mean(left_eye_loss)+torch.mean(right_eye_loss)+torch.mean(mouth_loss)


class MouthEyeProfierDisLoss(BaseLoss):
    def __init__(self, ignore_value=-1):
        super().__init__(LossName.MouthEyeFrontDisLoss)
        self.ignore_value = ignore_value

    def forward(self, output, targets):
        device = output.device
        batch_size = output.size(0)
        predictions = output.reshape((batch_size, -1))
        gt_data = targets.reshape((batch_size, -1)).to(device)
        weight_mask = torch.where(self.ignore_value,
                                  torch.zeros_like(targets),
                                  torch.ones_like(targets))
        # eye
        predictions_eye = ((predictions[:, 18*2+1] - predictions[:, 16*2+1]).mul(weight_mask[:, 18*2+1]).mul(weight_mask[:, 16*2+1]) +
                           (predictions[:, 19*2+1] - predictions[:, 15*2+1]).mul(weight_mask[:, 19*2+1]).mul(weight_mask[:, 15*2+1])) / 2

        targets_eye = ((gt_data[:, 18*2+1] - gt_data[:, 16*2+1]).mul(weight_mask[:, 18*2+1]).mul(weight_mask[:, 16*2+1]) +
                       (gt_data[:, 19*2+1] - gt_data[:, 15*2+1]).mul(weight_mask[:, 19*2+1]).mul(weight_mask[:, 15*2+1])) / 2
        t_eye = torch.abs(predictions_eye - targets_eye)
        # print('t_eye shape:', t_left_eye.size())
        eye_loss = torch.where(t_eye < 1, 0.5*t_eye.mul(t_eye), t_eye - 0.5)


        # mouth
        predictions_mouth = (predictions[:, 34*2+1] - predictions[:, 38*2+1]).mul(weight_mask[:, 34*2+1]).mul(weight_mask[:, 38*2+1])

        targets_mouth = (gt_data[:, 34*2+1] - gt_data[:, 38*2+1]).mul(weight_mask[:, 34*2+1]).mul(weight_mask[:, 38*2+1])
        t_mouth = torch.abs(predictions_mouth - targets_mouth)
        mouth_loss = torch.where(t_mouth < 1, 0.5*t_mouth.mul(t_mouth), t_mouth - 0.5)

        return torch.mean(eye_loss)+torch.mean(mouth_loss)
