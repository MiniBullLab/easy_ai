#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import cv2
import numpy as np
import random
import math
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess


class ImageDataSetProcess(BaseDataSetProcess):

    def __init__(self):
        super().__init__()

    def resize(self, src_image, dst_size, resize_type, **param):
        result = None
        if resize_type == 0:
            result = self.cv_image_resize(src_image, dst_size)
        elif resize_type == 1:
            pad_color = param['pad_color']
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            ratio, pad_size = self.get_square_size(src_size, dst_size)
            result = self.image_resize_square(src_image, ratio, pad_size,
                                              pad_color=pad_color)
        elif resize_type == -1:
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            # resize_ratio = dst_size[1] / src_image.shape[0]
            # resize_w = int(src_size[0] * resize_ratio)
            # dst_size = (resize_w, dst_size[1])
            # result = self.cv_image_resize(src_image, dst_size, interpolation="bilinear")
            ratio = src_size[0] / float(src_size[1])
            resize_w = int(math.ceil(dst_size[1] * ratio))
            if resize_w > dst_size[0]:
                resize_w = dst_size[0]
            dst_size = (resize_w, dst_size[1])
            result = self.cv_image_resize(src_image, dst_size, interpolation="bilinear")
        elif resize_type == -2:
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            resize_w, resize_h = self.get_short_size(src_size, dst_size)
            # print(resize_w, resize_h)
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            result = self.cv_image_resize(src_image, (int(resize_w), int(resize_h)))
        return result

    def inv_resize(self, src_size, dst_size, resize_type, image_data, **param):
        result = None
        if resize_type == 0:
            result = self.cv_image_resize(image_data, src_size)
        elif resize_type == 1:
            ratio, pad = self.get_square_size(src_size, dst_size)
            start_h = pad[1] // 2
            stop_h = dst_size[1] - (pad[1] - (pad[1] // 2))
            start_w = pad[0] // 2
            stop_w = dst_size[0] - (pad[0] - (pad[0] // 2))
            temp_result = image_data[start_h:stop_h, start_w:stop_w]
            result = self.cv_image_resize(temp_result, src_size)
        return result

    def normalize(self, input_data, normalize_type, **param):
        result = None
        if normalize_type == 0:
            result = self.image_normalize(input_data)
        elif normalize_type == 1:
            mean = param['mean']
            std = param['std']
            normaliza_image = self.image_normalize(input_data)
            result = self.standard_normalize(normaliza_image, mean, std)
        elif normalize_type == 2:
            temp_x = self.image_normalize(input_data)
            result = (temp_x - 0.5) / 0.5
        return result

    def numpy_transpose(self, images, dtype=np.float32):
        result = None
        if images is None:
            result = None
        elif images.ndim == 2:
            result = images[np.newaxis, :, :]
            result = np.ascontiguousarray(result, dtype=dtype)
        elif images.ndim == 3:
            image = images.transpose(2, 0, 1)
            result = np.ascontiguousarray(image, dtype=dtype)
        elif images.ndim == 4:
            img_all = images.transpose(0, 3, 1, 2)
            result = np.ascontiguousarray(img_all, dtype=dtype)
        return result

    def image_normalize(self, image):
        return image / 255.0

    def standard_normalize(self, input_data, mean, std):
        return (input_data - mean) / std

    def cv_image_resize(self, src_image, image_size, interpolation="nearest"):
        image = None
        if interpolation.strip() == "nearest":
            image = cv2.resize(src_image, image_size, interpolation=cv2.INTER_NEAREST)
        elif interpolation.strip() == "bilinear":
            image = cv2.resize(src_image, image_size, interpolation=cv2.INTER_LINEAR)
        return image

    def cv_image_color_convert(self, src_image, flag):
        result = None
        if flag == 0:
            result = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        elif flag == 1:
            result = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        return result

    def image_resize_square(self, src_image, ratio, pad_size, pad_color=(0, 0, 0)):
        shape = src_image.shape[:2]  # shape = [height, width]
        new_shape = (round(shape[0] * ratio), round(shape[1] * ratio))
        top = pad_size[1] // 2
        bottom = pad_size[1] - (pad_size[1] // 2)
        left = pad_size[0] // 2
        right = pad_size[0] - (pad_size[0] // 2)
        new_size = (new_shape[1], new_shape[0])
        image = self.cv_image_resize(src_image, new_size)
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=pad_color)
        return image

    def get_square_size(self, src_size, dst_size):
        # ratio  = old / new
        assert src_size[0] != 0 and src_size[1] != 0
        ratio = min(float(dst_size[0]) / src_size[0], float(dst_size[1]) / src_size[1])
        new_shape = (round(src_size[0] * ratio), round(src_size[1] * ratio))
        dw = dst_size[0] - new_shape[0]  # width padding
        dh = dst_size[1] - new_shape[1]  # height padding
        pad_size = (dw, dh)
        return ratio, pad_size

    def image_affine(self, src_image, matrix, border_value=250):
        result = None
        width = src_image.shape[1]
        height = src_image.shape[0]
        if src_image is not None:
            result = cv2.warpPerspective(src_image, matrix,
                                         dsize=(width, height),
                                         flags=cv2.INTER_NEAREST,
                                         borderValue=border_value)
        return result

    def affine_matrix(self, image_size, degrees=(-10, 10),
                      translate=(.1, .1), scale=(.9, 1.1),
                      shear=(-3, 3)):
        width = image_size[0]
        height = image_size[1]
        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[1] * width
        T[1, 2] = (random.random() * 2 - 1) * translate[0] * height

        # Shear
        S = np.eye(3)
        # x shear (deg)
        S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
        # y shear (deg)
        S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

        M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!
        return M, a

    def get_random_size(self, dst_size, scale=(0.5, 2)):
        # Multi-Scale
        min_size = min(dst_size)
        scale_range = [int(x * 10) for x in scale]
        width = int(random.choice(scale_range) / 10 * min_size)
        scale = float(dst_size[0]) / float(dst_size[1])
        height = int(round(float(width / scale) / 32.0) * 32)
        return width, height

    # def get_random_size(self, dst_size, max_rate=0.95, min_rate=0.5):
    #     rate = np.random.random() * (max_rate - min_rate) + min_rate
    #     width = int(dst_size[0] * rate)
    #     height = int(dst_size[1] * rate)
    #     return width, height

    def get_short_size(self, src_size, dst_size):
        short_size = min(dst_size)
        if min(src_size) < short_size:
            if src_size[1] < src_size[0]:
                ratio = float(short_size) / src_size[1]
            else:
                ratio = float(short_size) / src_size[0]
        else:
            ratio = 1.
        resize_h = int(src_size[0] * ratio)
        resize_w = int(src_size[1] * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        return resize_w, resize_h
