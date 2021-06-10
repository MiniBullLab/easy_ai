#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.augment.image_data_augment import ImageDataAugment


class RecTextDataAugment():

    def __init__(self):
        self.is_augment_hsv = True
        self.is_augment_affine = True
        self.image_augment = ImageDataAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        ocr_object = label.copy()
        ocr_object.clear_polygon()
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_affine:
            image, matrix = self.image_augment.augment_rotate(image, (-5, 5))
        return image, ocr_object
