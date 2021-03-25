#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from math import log10
from easyai.helper.average_meter import AverageMeter


class SuperResolutionPSNR():

    def __init__(self):
        self.epoch_avg_psnr = AverageMeter()

    def reset(self):
        self.epoch_avg_psnr.reset()

    def eval(self, loss):
        psnr = 10 * log10(1 / loss.item())
        self.epoch_avg_psnr.update(psnr, 1)

    def get_score(self):
        self.print_evaluation()
        return self.epoch_avg_psnr.avg

    def print_evaluation(self):
        print("Average psnr: {.5f}".format(self.epoch_avg_psnr.avg))
