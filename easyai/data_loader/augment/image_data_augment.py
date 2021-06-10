#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import random
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
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

    def random_line(self, src_image):
        """
            在图像增加一条简单的随机线
        """
        image = src_image[:]
        h = image.height
        w = image.width
        y0 = random.randint(h // 4, h * 3 // 4)
        y1 = np.clip(random.randint(-3, 3) + y0, 0, h - 1)
        color = random.randint(0, 30)
        cv2.line(image, (0, y0), (w - 1, y1), (color, color, color), 2)
        return image

    def random_compress(self, src_image, lower=5, upper=85):
        """
            随机压缩率，利用jpeg的有损压缩来增广
        """
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        image = src_image[:]
        param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(lower, upper)]
        img_encode = cv2.imencode('.jpeg', image, param)
        image = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
        return image

    def random_salt(self, src_image, rate=0.02):
        """
            随机椒盐噪音
        """
        image = Image.fromarray(src_image)
        num_noise = int(image.size[1] * image.size[0] * rate)
        # assert len(image.split()) == 1
        for k in range(num_noise):
            i = int(np.random.random() * image.size[1])
            j = int(np.random.random() * image.size[0])
            image.putpixel((j, i), int(np.random.random() * 255))
        return image

    def motion_blur(self, src_image, degree=5, angle=180):
        """
            随机运动模糊
        """
        image = Image.fromarray(src_image)
        angle = random.randint(0, angle)
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        image = image.filter(ImageFilter.Kernel(size=(degree, degree),
                                                kernel=motion_blur_kernel.reshape(-1)))
        return np.asarray(image)

    def random_contrast(self, src_image, lower=0.5, upper=1.5):
        """
            随机对比度
        """
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        image = Image.fromarray(src_image)
        contrast_enhance = ImageEnhance.Contrast(image)
        image = contrast_enhance.enhance(random.uniform(lower, upper))
        # bri = ImageEnhance.Brightness(image)
        # image = bri.enhance(random.uniform(lower, upper))
        return np.asarray(image)

    def random_color(self, src_image, lower=0.5, upper=1.5):
        """
            随机色彩平衡
       """
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        image = Image.fromarray(src_image)
        col = ImageEnhance.Color(image)
        image = col.enhance(random.uniform(lower, upper))
        return np.asarray(image)

    def random_sharpness(self, src_image, lower=0.5, upper=1.5):
        """
            随机锐度
       """
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        image = Image.fromarray(src_image)
        sha = ImageEnhance.Sharpness(image)
        image = sha.enhance(random.uniform(lower, upper))
        return np.asarray(image)

    def random_exposure(self, src_image, lower=5, upper=10):
        """
            随机区域曝光
        """
        assert upper >= lower, "upper must be >= lower."
        assert lower >= 0, "lower must be non-negative."
        image = src_image[:]
        h, w = image.shape[:2]
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        x1 = random.randint(x0, w)
        y1 = random.randint(y0, h)
        transparent_area = (x0, y0, x1, y1)
        mask = Image.new('L', (w, h), color=255)
        draw = ImageDraw.Draw(mask)
        mask = np.array(mask)
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
            mask = np.concatenate([mask, mask, mask], axis=2)
        draw.rectangle(transparent_area, fill=random.randint(150, 255))
        reflection_result = image + (255 - mask)
        reflection_result = np.clip(reflection_result, 0, 255)
        return reflection_result

    def random_crop(self, src_image, maxv=2):
        """
            随机抠图，并且抠图区域透视变换为原图大小
        """
        image = src_image[:]
        h, w = image.shape[:2]
        org = np.array([[0, np.random.randint(0, maxv)],
                        [w, np.random.randint(0, maxv)],
                        [0, h - np.random.randint(0, maxv)],
                        [w, h - np.random.randint(0, maxv)]], np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
        M = cv2.getPerspectiveTransform(org, dst)
        result = cv2.warpPerspective(image, M, (w, h))
        return result

    def inverse_color(self, src_image):
        image = Image.fromarray(src_image)
        if np.random.random() < 0.4:
            image = ImageOps.invert(image)
        return np.asarray(image)

    def random_stretch(self, src_image, max_rate=1.2, min_rate=0.8):
        """
            随机图像横向拉伸
        """
        image = src_image[:]
        w, h = image.size
        rate = np.random.random() * (max_rate - min_rate) + min_rate
        w2 = int(w * rate)
        result = cv2.resize(image, (w2, h))
        return result




