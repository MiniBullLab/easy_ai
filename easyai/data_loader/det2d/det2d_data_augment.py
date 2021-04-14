#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.augment.image_data_augment import ImageDataAugment
from easyai.data_loader.augment.box2d_augment import Box2dAugment


class DetectionDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.is_up_flip = False
        self.image_augment = ImageDataAugment()
        self.box_augemnt = Box2dAugment()

    def augment(self, image_rgb, labels):
        image = image_rgb[:]
        targets = labels[:]
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix, degree = self.image_augment.augment_affine(image)
            targets = self.labels_affine(targets, matrix, degree, image_size)
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            targets = self.labels_lr_flip(targets, is_lr, image_size)
        if self.is_up_flip:
            image, is_up = self.image_augment.augment_up_flip(image)
            targets = self.labels_up_flip(targets, is_up, image_size)
        return image, targets

    def labels_affine(self, labels, matrix, degree, image_size):
        targets = []
        for box in labels:
            result = self.box_augemnt.box_affine(box, matrix, degree, image_size)
            if result is not None:
                targets.append(result)
        return targets

    def labels_lr_flip(self, labels, is_lr, image_size):
        # left-right flip
        if is_lr:
            targets = []
            for box in labels:
                box = self.box_augemnt.box_lr_flip(box, image_size)
                targets.append(box)
            labels = targets
        return labels

    def labels_up_flip(self, labels, is_up, image_size):
        # up-down flip
        if is_up:
            targets = []
            for box in labels:
                box = self.box_augemnt.box_up_flip(box, image_size)
                targets.append(box)
            labels = targets
        return labels
