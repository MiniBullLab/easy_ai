#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import math
import random
import numpy as np
import torch
from easyai.data_loader.utility.base_data_loader import DataLoader
from easyai.data_loader.det2d.det2d_sample import DetectionSample
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.det2d.det2d_data_augment import DetectionDataAugment
from easyai.tools.sample_tool.create_detection_sample import CreateDetectionSample
from easyai.name_manager.dataloader_name import DataloaderName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_TRAIN_DATALOADER
from easyai.utility.logger import EasyLogger


@REGISTERED_TRAIN_DATALOADER.register_module(DataloaderName.Det2dTrainDataloader)
class Det2dTrainDataloader(DataLoader):

    def __init__(self, data_path, detect2d_class,
                 resize_type, normalize_type, mean=0, std=1,
                 batch_size=1, image_size=(768, 320),
                 data_channel=3, multi_scale=False,
                 is_augment=False, balanced_sample=False,
                 transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.detect2d_class = detect2d_class
        self.multi_scale = multi_scale
        self.is_augment = is_augment
        self.balanced_sample = balanced_sample
        self.batch_size = batch_size
        self.image_size = tuple(image_size)
        EasyLogger.debug("det2d class: {}".format(detect2d_class))

        if balanced_sample:
            create_sample = CreateDetectionSample()
            save_sample_dir, _ = os.path.split(data_path)
            create_sample.create_balance_sample(data_path,
                                                save_sample_dir,
                                                detect2d_class)
        self.detection_sample = DetectionSample(data_path,
                                                detect2d_class,
                                                balanced_sample)
        self.detection_sample.read_sample()

        self.dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                       mean, std, self.get_pad_color())
        self.dataset_augment = DetectionDataAugment()

        self.nF = self.detection_sample.get_sample_count()
        self.nB = math.ceil(self.nF / batch_size)  # number of batches

    def __iter__(self):
        self.count = -1
        self.detection_sample.shuffle_sample()
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration
        list_images = []
        list_labels = []

        class_index = self.get_random_class()
        start_index = self.detection_sample.get_sample_start_index(self.count,
                                                                   self.batch_size,
                                                                   class_index)
        dst_size = self.get_image_size()

        stop_index = start_index + self.batch_size
        for temp_index in range(start_index, stop_index):
            img_path, label_path = self.detection_sample.get_sample_path(temp_index, class_index)
            _, src_image = self.read_src_image(img_path)
            boxes = self.detection_sample.get_sample_boxes(label_path)
            image, labels = self.dataset_process.resize_dataset(src_image,
                                                                dst_size,
                                                                boxes,
                                                                self.detect2d_class)
            image, labels = self.dataset_augment.augment(image, labels)
            image = self.dataset_process.normalize_image(image)
            labels = self.dataset_process.normalize_labels(labels, dst_size)
            labels = self.dataset_process.change_outside_labels(labels)

            list_images.append(image)

            torch_labels = self.dataset_process.numpy_to_torch(labels, flag=0)
            list_labels.append(torch_labels)

        torch_images = torch.stack(list_images, dim=0)
        return {'image': torch_images, 'label': list_labels}

    def __len__(self):
        return self.nB  # number of batches

    def get_random_class(self):
        class_index = None
        if self.balanced_sample:
            class_index = np.random.randint(0, len(self.detect2d_class))
            EasyLogger.debug("loading labels {}".format(self.detect2d_class[class_index]))
        return class_index

    def get_image_size(self):
        if self.multi_scale:
            # Multi-Scale YOLO Training
            print("wrong code for MultiScale")
            width = random.choice(range(10, 20)) * 32  # 320 - 608 pixels
            scale = float(self.image_size[0]) / float(self.image_size[1])
            height = int(round(float(width / scale) / 32.0) * 32)
        else:
            # Fixed-Scale YOLO Training
            width = self.image_size[0]
            height = self.image_size[1]
        result_size = (width, height)
        return result_size
