#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.common.box2d_dataset_process import Box2dDataSetProcess


class DetectionDataSetProcess(Box2dDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)

    def resize_dataset(self, src_image, image_size, boxes, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_image, image_size)
        labels = self.resize_box(boxes, class_name, src_size, image_size)
        return image, labels

    def normalize_labels(self, labels, image_size):
        result = np.zeros((len(labels), 5), dtype=np.float32)
        for index, rect in enumerate(labels):
            class_id = rect.class_id
            x, y = rect.center()
            x /= image_size[0]
            y /= image_size[1]
            width = rect.width() / image_size[0]
            height = rect.height() / image_size[1]
            result[index, :] = np.array([class_id, x, y, width, height])
        return result

    def change_outside_labels(self, labels):
        delete_index = []
        # reject warped points outside of image (0.999 for the image boundary)
        for i, label in enumerate(labels):
            if label[2] + label[4] / 2 >= float(1):
                yoldH = label[2] - label[4] / 2
                label[2] = (yoldH + float(0.999)) / float(2)
                label[4] = float(0.999) - yoldH
            if label[1] + label[3] / 2 >= float(1):
                yoldW = label[1] - label[3] / 2
                label[1] = (yoldW + float(0.999)) / float(2)
                label[3] = float(0.999) - yoldW
            # filter the small object (w for label[3] in 1280 is limit to 6.8 pixel (6.8/1280=0.0053))
            if label[3] < 0.0053 or label[4] < 0.0055:
                # filter the small object (h for label[4] in 720 is limit to 4.0 pixel (4.0/1280=0.0053))
                delete_index.append(i)

        labels = np.delete(labels, delete_index, axis=0)
        return labels

    def normalize_clip_labels(self, labels, image_size, clip=True, eps=1E-3):
        result = np.zeros((len(labels), 5), dtype=np.float32)
        for index, rect in enumerate(labels):
            class_id = rect.class_id
            x1 = rect.min_corner.x
            y1 = rect.min_corner.y
            x2 = rect.max_corner.x
            y2 = rect.max_corner.y
            result[index, :] = np.array([class_id, x1, y1, x2, y2])
        if clip:
            self.clip_coords(result, (image_size[0] - eps, image_size[1] - eps))  # warning: inplace clip
        y = np.copy(result)
        y[:, 1] = ((result[:, 1] + result[:, 3]) / 2) / image_size[0]  # x center
        y[:, 2] = ((result[:, 2] + result[:, 4]) / 2) / image_size[1]  # y center
        y[:, 3] = (result[:, 3] - result[:, 1]) / image_size[0]  # width
        y[:, 4] = (result[:, 4] - result[:, 2]) / image_size[1]  # height
        return y

    def clip_coords(self, boxes, shape):
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # x1, x2
        boxes[:, [2, 4]] = boxes[:, [2, 4]].clip(0, shape[1])  # y1, y2
