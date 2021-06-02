#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import random
import numpy as np
from skimage.util import random_noise
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess


class ImageDataAugment():

    def __init__(self):
        self.dataset_process = ImageDataSetProcess()
        self.border_value = (0.0, 0.0, 0.0)

    def augment_affine(self, src_image, degrees=(-15, 15),
                       translate=(0.0, 0.0), scale=(1.0, 1.0), shear=(-3, 3)):
        image_size = (src_image.shape[1], src_image.shape[0])
        matrix, degree = self.dataset_process.affine_matrix(image_size,
                                                            degrees=degrees,
                                                            translate=translate,
                                                            scale=scale,
                                                            shear=shear)
        image = self.dataset_process.image_affine(src_image, matrix,
                                                  border_value=self.border_value)
        return image, matrix, degree

    def augment_rotate(self, src_image, degrees=(-10, 10)):
        image_size = (src_image.shape[1], src_image.shape[0])
        angle = np.random.uniform(degrees[0], degrees[1])
        # 构造仿射矩阵
        matrix = cv2.getRotationMatrix2D((image_size[0] * 0.5, image_size[1] * 0.5),
                                         angle, 1)
        # 仿射变换
        image = cv2.warpAffine(src_image, matrix, image_size,
                               flags=cv2.INTER_LANCZOS4)
        return image, matrix

    def augment_lr_flip(self, src_image):
        image = src_image[:]
        is_lr = False
        if random.random() > 0.5:
            image = np.fliplr(image)
            is_lr = True
        return image, is_lr

    def augment_up_flip(self, src_image):
        # random up-down flip
        image = src_image[:]
        is_up = False
        if random.random() > 0.5:
            image = np.flipud(image)
            is_up = True
        return image, is_up

    def augment_hsv(self, rgb_image):
        # SV augmentation by 50%
        fraction = 0.50
        img_hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        a = (random.random() * 2 - 1) * fraction + 1
        S *= a
        if a > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        a = (random.random() * 2 - 1) * fraction + 1
        V *= a
        if a > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return result

    def gaussian_blur(self, src_image):
        # random gaussian blur
        image_size = (src_image.shape[1], src_image.shape[0])
        image = src_image[:]
        if image_size[0] >= 90 and random.randint(0, 1) == 0:
            image = cv2.GaussianBlur(image, (5, 5), 1)
        return image

    def random_noise(self, src_image):
        image = src_image[:]
        if random.random() > 0.5:
            image = (random_noise(image, mode='gaussian',
                                  clip=True) * 255).astype(image.dtype)
        return image
