#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np


class KNN():

    def __init__(self, k=3):
        self.k = k

    def __call__(self, x, y):
        dist = np.zeros((x.shape[0], y.shape[0]), dtype=np.float32)
        for i in range(x.shape[0]):
            dist[i, :] = np.sum((y - x[i, :]) ** 2, axis=1)
        dist = np.sqrt(dist)
        # result_index = np.argsort(dist, axis=1)[:, :self.k]
        result = np.sort(dist, axis=1)[:, :self.k]
        return result
