#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import numpy as np
from easyai.helper.dataType import Rect2D
from easyai.data_loader.utility.base_dataset_process import BaseDataSetProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess


class DetectionDataSetProcess(BaseDataSetProcess):

    def __init__(self, image_pad_color=(0, 0, 0)):
        super().__init__()
        self.dataset_process = ImageDataSetProcess()
        self.image_pad_color = image_pad_color

    def normalize_image(self, src_image):
        image = self.dataset_process.image_normalize(src_image)
        image = self.dataset_process.numpy_transpose(image)
        return image

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

    def resize_dataset(self, src_image, image_size, boxes, class_name):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        ratio, pad_size = self.dataset_process.get_square_size(src_size, image_size)
        image = self.dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                         color=self.image_pad_color)
        labels = self.resize_labels(boxes, class_name, ratio, pad_size)
        return image, labels

    def resize_src_image(self, src_image, image_size):
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        ratio, pad_size = self.dataset_process.get_square_size(src_size, image_size)
        image = self.dataset_process.image_resize_square(src_image, ratio, pad_size,
                                                         color=self.image_pad_color)
        return image

    def resize_labels(self, boxes, class_name, ratio, pad_size):
        labels = []
        for box in boxes:
            if box.name in class_name:
                rect = Rect2D()
                rect.class_id = class_name.index(box.name)
                rect.min_corner.x = ratio * box.min_corner.x + pad_size[0] // 2
                rect.min_corner.y = ratio * box.min_corner.y + pad_size[1] // 2
                rect.max_corner.x = ratio * box.max_corner.x + pad_size[0] // 2
                rect.max_corner.y = ratio * box.max_corner.y + pad_size[1] // 2
                labels.append(rect)
        return labels

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
