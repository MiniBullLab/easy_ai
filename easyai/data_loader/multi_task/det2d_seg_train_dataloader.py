#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import math
import random
import numpy as np
import torch
from easyai.helper.json_process import JsonProcess
from easyai.data_loader.utility.base_data_loader import DataLoader
from easyai.data_loader.multi_task.multi_task_sample import MultiTaskSample
from easyai.data_loader.multi_task.det2d_seg_data_augment import Det2dSegDataAugment
from easyai.data_loader.det2d.det2d_dataset_process import DetectionDataSetProcess
from easyai.data_loader.seg.segment_dataset_process import SegmentDatasetProcess
from easyai.tools.sample_tool.create_detection_sample import CreateDetectionSample
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.name_manager.dataloader_name import DataloaderName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_TRAIN_DATALOADER


@REGISTERED_TRAIN_DATALOADER.register_module(DataloaderName.Det2dSegTrainDataloader)
class Det2dSegTrainDataloader(DataLoader):

    def __init__(self, data_path, detect2d_class, seg_class_name,
                 seg_label_type, resize_type, normalize_type, mean=0, std=1,
                 batch_size=1, image_size=(768, 320), data_channel=3,
                 multi_scale=False, is_augment=False, balanced_sample=False,
                 transform_func=None):
        super().__init__(data_path, data_channel, transform_func)
        self.detect2d_class = detect2d_class
        self.seg_number_class = len(seg_class_name)
        self.seg_label_type = seg_label_type
        self.multi_scale = multi_scale
        self.is_augment = is_augment
        self.balanced_sample = balanced_sample
        self.batch_size = batch_size
        self.image_size = image_size

        self.seg_label_converter = ConvertSegmentionLable()

        if balanced_sample:
            create_sample = CreateDetectionSample()
            save_sample_dir, _ = os.path.split(data_path)
            create_sample.createBalanceSample(data_path,
                                              save_sample_dir,
                                              detect2d_class)
        self.multi_task_sample = MultiTaskSample(data_path,
                                                 detect2d_class,
                                                 balanced_sample)
        self.multi_task_sample.read_sample()
        self.json_process = JsonProcess()

        self.det2d_dataset_process = DetectionDataSetProcess(resize_type, normalize_type,
                                                             mean, std, self.get_pad_color())
        self.seg_dataset_process = SegmentDatasetProcess(resize_type, normalize_type,
                                                         mean, std, self.get_pad_color())
        self.dataset_augment = Det2dSegDataAugment()

        self.nF = self.multi_task_sample.get_sample_count()
        self.nB = math.ceil(self.nF / batch_size)  # number of batches

    def __iter__(self):
        self.count = -1
        self.multi_task_sample.shuffle_sample()
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nB:
            raise StopIteration

        list_images = []
        list_boxes = []
        list_segments = []

        class_index = self.get_random_class()
        start_index = self.multi_task_sample.get_sample_start_index(self.count,
                                                                    self.batch_size,
                                                                    class_index)
        dst_size = self.get_image_size()

        stop_index = start_index + self.batch_size
        for temp_index in range(start_index, stop_index):
            img_path, label_path, segment_path = self.multi_task_sample.get_sample_path(temp_index,
                                                                                        class_index)
            _, src_image = self.read_src_image(img_path)
            _, boxes = self.json_process.parse_rect_data(label_path)
            segment_label = self.read_seg_label_image(segment_path)

            src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
            image = self.det2d_dataset_process.resize_image(src_size, dst_size)

            boxes = self.det2d_dataset_process.resize_box(boxes, self.detect2d_class, src_size, dst_size)
            segment_label = self.seg_dataset_process.resize_lable(segment_label, src_size, dst_size)
            if self.is_augment:
                image, boxes, segments = self.dataset_augment.augment(image, boxes, segment_label)

            image = self.det2d_dataset_process.normalize_image(image)
            boxes = self.det2d_dataset_process.normalize_labels(boxes, dst_size)

            boxes = self.det2d_dataset_process.change_outside_labels(boxes)
            segment_label = self.seg_dataset_process.change_label(segment_label, self.seg_number_class)

            torch_boxes = self.det2d_dataset_process.numpy_to_torch(boxes, flag=0)
            torch_seg_label = self.det2d_dataset_process.numpy_to_torch(segment_label, flag=0).long()

            list_images.append(image)
            list_boxes.append(torch_boxes)
            list_segments.append(torch_seg_label)

        torch_images = torch.stack(list_images, dim=0)
        torch_segments = torch.stack(list_segments, dim=0)

        return torch_images, list_boxes, torch_segments

    def __len__(self):
        return self.nB  # number of batches

    def get_random_class(self):
        class_index = None
        if self.balanced_sample:
            class_index = np.random.randint(0, len(self.detect2d_class))
            print("loading labels {}".format(self.detect2d_class[class_index]))
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

    def read_seg_label_image(self, label_path):
        if self.seg_label_type == 0:
            mask = self.image_process.read_gray_image(label_path)
        else:
            mask = self.seg_label_converter.process_segment_label(label_path,
                                                                  self.seg_label_type,
                                                                  self.seg_number_class)
        return mask
