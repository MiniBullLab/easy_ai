#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np


def detection2d_labels_weights(labels, class_number):
    # Get class weights (inverse frequency) from training labels
    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=class_number)  # occurences per class
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights


def detection2d_image_weights(labels, class_number, class_weights):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=class_number) for i in range(n)])
    image_weights = (class_weights.reshape(1, class_number) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights
