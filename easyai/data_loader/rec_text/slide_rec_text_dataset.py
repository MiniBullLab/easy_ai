#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.torch_data_loader import TorchDataLoader
from easyai.data_loader.rec_text.rec_text_sample import RecTextSample
from easyai.data_loader.rec_text.rec_text_dataset_process import RecTextDataSetProcess
from easyai.data_loader.rec_text.rec_text_augment import RecTextDataAugment
from easyai.name_manager.dataloader_name import DatasetName
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.utility.logger import EasyLogger


@REGISTERED_DATASET.register_module(DatasetName.SlideRecTextDataSet)
class SlideRecTextDataSet(TorchDataLoader):

    def __init__(self, data_path, char_path, max_text_length, language,
                 resize_type, normalize_type, mean=0, std=1,
                 image_size=(416, 416), data_channel=3, windows=(24, 32, 40),
                 step=4, is_augment=False):
        super().__init__(data_path, data_channel)
        self.char_path = char_path
        self.max_text_length = max_text_length
        self.language = language
        self.image_size = tuple(image_size)
        self.windows = windows
        self.step = step
        self.is_augment = is_augment

        self.dataset_process = RecTextDataSetProcess(resize_type, normalize_type,
                                                     mean, std, self.get_pad_color())
        EasyLogger.debug(char_path)
        character = self.dataset_process.read_character(char_path)
        self.text_sample = RecTextSample(data_path, language)
        self.text_sample.read_text_sample(character,
                                          self.max_text_length)

        self.dataset_augment = RecTextDataAugment()

    def __getitem__(self, index):
        img_path, label = self.text_sample.get_sample_path(index)
        _, src_image = self.read_src_image(img_path)
        if self.is_augment:
            image, label = self.dataset_augment.augment(src_image, label)
        else:
            image = src_image[:]
        image = self.dataset_process.resize_image(image, self.image_size)
        # image = self.dataset_process.padding_images(image, self.image_size)
        image = self.dataset_process.normalize_image(image)
        image = self.dataset_process.width_pad_images(image, self.image_size[0], 2)
        image = self.dataset_process.slide_image(image, self.windows, self.step)
        label = self.dataset_process.normalize_label(label)
        result_data = {'image': image}
        result_data.update(label)
        return result_data

    def __len__(self):
        return self.text_sample.get_sample_count()

