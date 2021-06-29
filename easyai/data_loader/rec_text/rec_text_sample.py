#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import lmdb
from easyai.helper.data_structure import OCRObject
from easyai.helper.image_process import ImageProcess
from easyai.data_loader.utility.base_detection_sample import BaseDetectionSample
from easyai.data_loader.utility.base_classify_sample import BaseClassifySample
from easyai.utility.logger import EasyLogger


class RecTextSample(BaseDetectionSample):

    def __init__(self, train_path, language):
        super().__init__()
        self.image_process = ImageProcess()
        self.train_path = train_path
        self.language = language
        self.max_length = 0
        self.image_and_ocr_list = []
        self.sample_count = 0

        self.lmdb_env = None

    def read_sample(self, char_list):
        image_and_label_list = self.get_image_and_label_list(self.train_path)
        self.image_and_ocr_list = self.get_image_and_ocr_list(image_and_label_list)
        filtering_result = []
        for image_path, ocr_object in self.image_and_ocr_list:
            for char in ocr_object.object_text:
                if char in char_list:
                    filtering_result.append((image_path, ocr_object))
                    break
        self.image_and_ocr_list = filtering_result

        self.sample_count = self.get_sample_count()

    def read_lmdb_sample(self, char_list):
        self.lmdb_env = lmdb.open(self.train_path, max_readers=32,
                                  readonly=True, lock=False,
                                  readahead=False, meminit=False)
        if not self.lmdb_env:
            EasyLogger.warn('cannot create lmdb from %s' % self.train_path)
        else:
            with self.lmdb_env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                for index in range(nSamples):
                    label_key = 'label_%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    if True in [c not in char_list for c in label]:
                        continue
                    self.image_and_ocr_list.append(index)
        self.sample_count = self.get_sample_count()

    def read_text_sample(self, char_list):
        self.image_and_ocr_list = []
        sample_process = BaseClassifySample()
        image_and_text_list = sample_process.get_image_and_text_list(self.train_path)
        for image_path, text_data in image_and_text_list:
            if not text_data.strip():
                continue
            if True in [c not in char_list for c in text_data.strip()]:
                continue
            ocr_object = OCRObject()
            ocr_object.set_text(text_data)
            self.image_and_ocr_list.append((image_path, ocr_object))
        self.sample_count = self.get_sample_count()

    def get_sample_path(self, index):
        temp_index = index % self.sample_count
        img_path, ocr_object = self.image_and_ocr_list[temp_index]
        return img_path, ocr_object

    def get_lmdb_sample(self, index):
        temp_index = index % self.sample_count
        index = self.image_and_ocr_list.append(temp_index)
        with self.lmdb_env.begin(write=False) as txn:
            label_key = 'label_%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image_%09d'.encode() % index
            img_buffer = txn.get(img_key)
            src_image = self.image_process.read_raw_buf(img_buffer)
            ocr_object = OCRObject()
            ocr_object.set_text(label)
        return src_image, ocr_object

    def get_sample_count(self):
        return len(self.image_and_ocr_list)

    def get_image_and_ocr_list(self, image_and_label_list):
        result = []
        for image_path, label_path in image_and_label_list:
            _, ocr_objects = self.json_process.parse_ocr_data(label_path)
            for ocr in ocr_objects:
                if ocr.language.strip() in self.language:
                    self.max_length = max(len(ocr.object_text), self.max_length)
                    result.append((image_path, ocr))
        return result

