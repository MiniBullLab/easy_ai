#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.rec_text.rec_text_sample import RecTextSample
from easyai.data_loader.rec_text.rec_text_dataset_process import RecTextDataSetProcess
from easyai.data_loader.rec_text.rec_text_augment import RecTextDataAugment


class RecTextDataSet(TorchDataLoader):

    def __init__(self, data_path, char_path, language,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3,
                 is_augment=False):
        super().__init__(data_channel)
        self.char_path = char_path
        self.language = language
        self.image_size = image_size
        self.is_augment = is_augment
        self.text_sample = RecTextSample(data_path, language)
        self.text_sample.read_sample()

        self.dataset_process = RecTextDataSetProcess(resize_type, normalize_type,
                                                     mean, std, self.get_pad_color())

        self.dataset_augment = RecTextDataAugment()

    def __getitem__(self, index):
        img_path, label = self.text_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        image = self.dataset_process.get_rotate_crop_image(src_image,
                                                           label.get_polygon())
        image = self.dataset_process.resize_image(image, self.image_size)
        if self.is_augment:
            image, label = self.dataset_augment.augment(image, label)

        image = self.dataset_process.normalize_image(image)
        label = self.dataset_process.normalize_label(label)
        return image, label

    def __len__(self):
        return self.text_sample.get_sample_count()
