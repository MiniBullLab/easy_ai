#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import random
import numpy as np
from easyai.data_loader.augment.image_data_augment import ImageDataAugment
from easyai.data_loader.augment.text_image_augment import TextImageAugment


class RecTextDataAugment():

    def __init__(self):
        self.probability = 0.4
        self.is_hsv = True
        self.is_rotate = False
        self.is_reverse = True
        self.is_crop = True
        self.is_noise = True
        self.is_jitter = True
        self.is_blur = True
        self.is_distort = True
        self.is_stretch = True
        self.is_perspective = True
        self.image_augment = ImageDataAugment()
        self.text_augment = TextImageAugment()

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        ocr_object = label.copy()
        ocr_object.clear_polygon()
        if self.is_hsv and (random.random() <= self.probability):
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_rotate and (random.random() <= self.probability):
            image, matrix = self.image_augment.augment_rotate(image, (-5, 5))
        if self.is_distort and (random.random() <= self.probability):
            image = self.text_augment.distort(image, random.randint(3, 6))
        if self.is_stretch and (random.random() <= self.probability):
            image = self.text_augment.stretch(image, random.randint(3, 6))
        if self.is_perspective and (random.random() <= self.probability):
            image = self.text_augment.perspective(image)
        if self.is_crop and (random.random() <= self.probability):
            image = self.get_crop(image)
        if self.is_blur and (random.random() <= self.probability):
            image = self.image_augment.gaussian_blur(image)
        if self.is_jitter and (random.random() <= self.probability):
            image = self.jitter(image)
        if self.is_noise and (random.random() <= self.probability):
            image = self.add_gasuss_noise(image)
        if self.is_reverse and (random.random() <= self.probability):
            image = 255 - image
        return image, ocr_object

    def get_crop(self, image):
        """
        random crop
        """
        h, w, _ = image.shape
        top_min = 1
        top_max = 8
        top_crop = int(random.randint(top_min, top_max))
        top_crop = min(top_crop, h - 1)
        crop_img = image.copy()
        ratio = random.randint(0, 1)
        if ratio:
            crop_img = crop_img[top_crop:h, :, :]
        else:
            crop_img = crop_img[0:h - top_crop, :, :]
        return crop_img

    def add_gasuss_noise(self, image, mean=0, var=0.1):
        """
        Gasuss noise
        """

        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + 0.5 * noise
        out = np.clip(out, 0, 255)
        out = np.uint8(out)
        return out

    def jitter(self, img):
        """
        jitter
        """
        w, h, _ = img.shape
        if h > 10 and w > 10:
            thres = min(w, h)
            s = int(random.random() * thres * 0.01)
            src_img = img.copy()
            for i in range(s):
                img[i:, i:, :] = src_img[:w - i, :h - i, :]
            return img
        else:
            return img
