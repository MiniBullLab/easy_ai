#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
import cv2
import random
import numpy as np
from easyai.helper.data_structure import Rect2D
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_data_augment import DetectionDataAugment
from easyai.data_loader.augment.albumentations import Albumentations
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.visualization.utility.image_drawing import ImageDrawing
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger


@REGISTERED_DATASET.register_module(DatasetName.MosaicDet2dDataset)
class MosaicDet2dDataset(TorchDataLoader):

    def __init__(self, data_path, detect2d_class,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(640, 640), data_channel=3, is_augment=False,
                 transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.image_size = tuple(image_size)
        self.detect2d_class = detect2d_class
        EasyLogger.debug("det2d class: {}".format(detect2d_class))
        self.detection_sample = DetectionSample(data_path,
                                                detect2d_class,
                                                False)
        self.detection_sample.read_sample()
        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())

        self.mosaic_border = [-image_size[0] // 2, -image_size[0] // 2]
        self.indices = range(self.detection_sample.get_sample_count())
        self.is_augment = is_augment

        self.dataset_augment = DetectionDataAugment(is_augment_affine=False)

        self.number = 0
        self.drawing = ImageDrawing()

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        if self.is_augment:
            image, labels = self.load_mosaic(index)
            image, targets = self.dataset_augment.random_perspective(image, labels,
                                                                     self.mosaic_border)
            labels = []
            for value in targets:
                rect = Rect2D()
                rect.class_id = value[0]
                rect.min_corner.x = value[1]
                rect.min_corner.y = value[2]
                rect.max_corner.x = value[3]
                rect.max_corner.y = value[4]
                labels.append(rect)
            image, labels = self.dataset_augment.augment(image, labels)
            # print(img_path)
            # self.drawing.draw_rect2d(image, labels)
            # self.drawing.save_image(image, "img_%d.png" % self.number)
            # self.number += 1
            src_size = np.array([0, 0])  # [width, height]
        else:
            _, src_image = self.read_src_image(img_path)
            boxes = self.detection_sample.get_sample_boxes(label_path)
            image, labels = self.dataset_process.resize_dataset(src_image,
                                                                self.image_size,
                                                                boxes,
                                                                self.detect2d_class)
            src_size = np.array([src_image.shape[1], src_image.shape[0]])  # [width, height]
        image = self.dataset_process.normalize_image(image)
        labels = self.dataset_process.normalize_clip_labels(labels, self.image_size)
        nl = len(labels)  # number of labels
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        src_size = self.dataset_process.numpy_to_torch(src_size, flag=0)
        return {'image': image, 'label': labels_out,
                'image_path': img_path, "src_size": src_size}

    def __len__(self):
        return self.detection_sample.get_sample_count()

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.image_size[0]
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, temp_index in enumerate(indices):
            # Load image
            img_path, label_path = self.detection_sample.get_sample_path(temp_index)
            _, src_image = self.read_src_image(img_path)
            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            r = self.image_size[0] / max(src_size)  # ratio
            if r != 1:  # if sizes are not equal
                image = cv2.resize(src_image, (int(src_size[0] * r), int(src_size[1] * r)),
                                   interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
            else:
                image = src_image
            h, w = image.shape[0], image.shape[1]

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, image.shape[2]), 0, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            boxes = self.detection_sample.get_sample_boxes(label_path)
            labels = []
            for box in boxes:
                rect = Rect2D()
                rect.class_id = self.detect2d_class.index(box.name)
                x1 = r * box.min_corner.x + padw
                y1 = r * box.min_corner.y + padh
                x2 = r * box.max_corner.x + padw
                y2 = r * box.max_corner.y + padh
                x1 = min(max(x1, 0), 2 * s)
                y1 = min(max(y1, 0), 2 * s)
                x2 = min(max(x2, 0), 2 * s)
                y2 = min(max(y2, 0), 2 * s)
                rect.min_corner.x = x1
                rect.min_corner.y = y1
                rect.max_corner.x = x2
                rect.max_corner.y = y2
                labels.append(rect)
            labels4.extend(labels)
        return img4, labels4

