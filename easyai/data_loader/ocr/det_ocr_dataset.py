#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.ocr.ocr_sample import OCRSample
from easyai.data_loader.ocr.ocr_dataset_process import OCRDataSetProcess
from easyai.data_loader.ocr.ocr_augment import OCRDataAugment
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger


@REGISTERED_DATASET.register_module(DatasetName.DetOCRDataSet)
class DetOCRDataSet(TorchDataLoader):

    def __init__(self, data_path, resize_type, normalize_type,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 image_size=(640, 640), data_channel=3,
                 is_augment=False, transform_func=None, language=""):
        super().__init__(data_path, data_channel, transform_func)
        self.language = language
        self.image_size = tuple(image_size)
        self.is_augment = is_augment
        self.expand_ratio = (1.0, 1.0)

        self.dataset_process = OCRDataSetProcess(resize_type, normalize_type,
                                                 mean, std, self.get_pad_color())
        self.ocr_sample = OCRSample(data_path, language)
        self.ocr_sample.read_sample()

        self.dataset_augment = OCRDataAugment(self.image_size)

    def __getitem__(self, index):
        img_path, ocr_objects = self.ocr_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        if self.is_augment:
            image, labels = self.dataset_augment.augment(src_image, ocr_objects)
        else:
            image, labels = self.dataset_process.resize_dataset(src_image, self.image_size,
                                                                ocr_objects)
        image, labels = self.dataset_process.normalize_dataset(image, labels)
        result_data = {'image': image}
        result_data.update(labels)
        return result_data

    def __len__(self):
        return self.text_sample.get_sample_count()

