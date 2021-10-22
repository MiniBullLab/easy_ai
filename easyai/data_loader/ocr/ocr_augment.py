#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
import imgaug
import imgaug.augmenters as iaa
from easyai.helper.data_structure import Point2d
from easyai.data_loader.augment.image_data_augment import ImageDataAugment
from easyai.data_loader.augment.east_random_crop import EastRandomCropData
from easyai.utility.logger import EasyLogger


class OCRDataAugment():

    def __init__(self, image_size):
        self.is_augment_hsv = False
        self.is_augment_others = False
        self.is_crop = True
        self.image_size = image_size
        self.image_augment = ImageDataAugment()
        self.random_crop = EastRandomCropData(size=self.image_size)

    def augment(self, image_rgb, ocr_objects):
        image = image_rgb[:]
        labels = ocr_objects[:]
        src_size = (image_rgb.shape[1], image_rgb.shape[0])
        if self.is_augment_hsv:
            image = self.image_augment.augment_hsv(image_rgb)
        if self.is_augment_others:
            image, labels = self.other_augment(image, ocr_objects)
        if self.is_crop:
            image, labels = self.random_crop.ocr_crop(image, labels)
            # EasyLogger.debug("train image size(W, H): %d, %d" % (image.shape[1],
            #                                                      image.shape[0]))
        return image, labels

    def label_affine(self, points, matrix, image_size):
        for point in points:
            if point.x < 0 or point.y < 0:
                continue
            new_pt = np.array([point.x, point.y, 1.]).T
            new_pt = np.dot(matrix, new_pt)
            if new_pt[0] < 0:
                point.x = 0
            elif new_pt[0] > image_size[0] - 1:
                point.x = image_size[0] - 1
            else:
                point.x = new_pt[0]
            if new_pt[1] < 0:
                point.y = 0
            elif new_pt[1] > image_size[1] - 1:
                point.y = image_size[1] - 1
            else:
                point.y = new_pt[1]
        return points

    def other_augment(self, image_rgb, ocr_objects):
        labels = []
        shape = image_rgb.shape
        fliplr_augment = iaa.Fliplr(p=0.5)
        affine_augment = iaa.Affine(rotate=[-10, 10])
        resize_augment = iaa.Resize(size=[0.5, 3.0])
        aug = fliplr_augment.to_deterministic()
        image = aug.augment_image(image_rgb)
        for ocr in ocr_objects:
            temp_ocr = ocr.copy()
            temp_poly = np.array([[p.x, p.y] for p in ocr.get_polygon()])
            temp_ocr.clear_polygon()
            for point in self.may_augment_poly(aug, shape, temp_poly):
                temp_ocr.add_point(point)
            labels.append(temp_ocr)
        aug = affine_augment.to_deterministic()
        image = aug.augment_image(image)
        for ocr in labels:
            temp_poly = np.array([[p.x, p.y] for p in ocr.get_polygon()])
            ocr.clear_polygon()
            for point in self.may_augment_poly(aug, shape, temp_poly):
                point.x = min(max(0, point.x), image.shape[1] - 1)
                point.y = min(max(0, point.y), image.shape[0] - 1)
                ocr.add_point(point)
        aug = resize_augment.to_deterministic()
        image = aug.augment_image(image)
        for ocr in labels:
            temp_poly = np.array([[p.x, p.y] for p in ocr.get_polygon()])
            ocr.clear_polygon()
            for point in self.may_augment_poly(aug, shape, temp_poly):
                ocr.add_point(point)
        return image, labels

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [Point2d(p.x, p.y) for p in keypoints]
        return poly

