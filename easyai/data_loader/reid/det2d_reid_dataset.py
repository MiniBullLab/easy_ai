#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_data_augment import DetectionDataAugment
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.name_manager.dataloader_name import DatasetName
from easyai.visualization.utility.image_drawing import ImageDrawing
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger


@REGISTERED_DATASET.register_module(DatasetName.Det2dReidDataset)
class Det2dReidDataset(TorchDataLoader):

    def __init__(self, data_path, detect2d_class,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(640, 640), data_channel=3, is_augment=False,
                 transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.image_size = tuple(image_size)
        self.detect2d_class = detect2d_class
        EasyLogger.debug("det2d class: {}".format(detect2d_class))
        self.detection_sample = DetectionSample(data_path,
                                                detect2d_class)
        self.detection_sample.read_tracking_sample()

        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())
        self.is_augment = is_augment

        self.dataset_augment = DetectionDataAugment()

        self.number = 0
        self.drawing = ImageDrawing()

    def __getitem__(self, index):
        img_path, label_path = self.detection_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        boxes = self.detection_sample.get_sample_boxes(label_path)

        image, labels = self.dataset_process.resize_dataset(src_image,
                                                            self.image_size,
                                                            boxes,
                                                            self.detect2d_class)
        if self.is_augment:
            image, labels = self.dataset_augment.augment(image, labels)
        # print(img_path, len(labels))
        # self.drawing.draw_tracking_objects(image, labels)
        # self.drawing.save_image(image, "img_%d.png" % self.number)
        # self.number += 1

        src_size = np.array([src_image.shape[1], src_image.shape[0]])  # [width, height]
        image = self.dataset_process.normalize_image(image)
        labels = self.dataset_process.normalize_tracking_labels(labels, self.image_size)
        # torch_labels = self.dataset_process.numpy_to_torch(labels, flag=0)
        src_size = self.dataset_process.numpy_to_torch(src_size, flag=0)
        return {'image': image, 'label': labels,
                'image_path': img_path, "src_size": src_size}

    def __len__(self):
        return self.detection_sample.get_sample_count()
