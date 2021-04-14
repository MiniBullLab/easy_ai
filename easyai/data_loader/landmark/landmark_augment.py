#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.pose2d.pose2d_augment import Pose2dDataAugment


class LandmarkDataAugment(Pose2dDataAugment):

    def __init__(self):
        super().__init__()
        self.is_augment_affine = True
        self.is_lr_flip = True
        self.is_blur = True
        self.flip_pairs = [[0, 16],  [1, 15],  [2, 14],  [3, 13],  [4, 12],  [5, 11],
                           [6, 10],  [7, 9],   [8, 8],   [9, 7],   [10, 6],  [11, 5],
                           [12, 4],  [13, 3],  [14, 2],  [15, 1],  [16, 0],  [17, 26],
                           [18, 25], [19, 24], [20, 23], [21, 22], [22, 21], [23, 20],
                           [24, 19], [25, 18], [26, 17], [27, 27], [28, 28], [29, 29],
                           [30, 30], [31, 35], [32, 34], [33, 33], [34, 32], [35, 31],
                           [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                           [42, 39], [43, 38], [44, 37], [45, 36], [46, 41], [47, 40],
                           [48, 54], [49, 53], [50, 52], [51, 51], [52, 50], [53, 49],
                           [54, 48], [55, 59], [56, 58], [57, 57], [58, 56], [59, 55],
                           [60, 64], [61, 63], [62, 62], [63, 61], [64, 60], [65, 67],
                           [66, 66], [67, 65]]

    def augment(self, image_rgb, label):
        image = image_rgb[:]
        keypoint = label.copy()
        keypoint.clear_key_points()
        image_size = (image_rgb.shape[1], image_rgb.shape[0])
        box = label.get_rect2d()
        flags = label.get_key_points_flag()
        points = label.get_key_points()
        if self.is_blur:
            image = self.image_augment.gaussian_blur(image_rgb)
        if self.is_augment_affine:
            copy_image = image[:]
            image, matrix, degree = self.image_augment.augment_affine(image)
            temp_box = self.box_augment.box_affine(box, matrix, degree, image_size)
            if temp_box is not None:
                box = temp_box
                points = self.label_affine(points, matrix, image_size)
            else:
                image = copy_image
        if self.is_lr_flip:
            image, is_lr = self.image_augment.augment_lr_flip(image)
            if is_lr:
                box = self.box_augment.box_lr_flip(box, image_size)
                flags = self.direction_cls_lr_flip(flags)
            points = self.label_lr_flip(points, is_lr, image_size)
        for point in points:
            keypoint.add_key_points(point)
        keypoint.set_rect2d(box)
        keypoint.set_key_points_flag(flags)
        return image, keypoint

    def direction_cls_lr_flip(self, flags):
        assert len(flags) > 0
        direction_cls = flags[0]
        if direction_cls == 1:
            direction_cls = 2
        elif direction_cls == 2:
            direction_cls = 1
        return direction_cls
