#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess
from easyai.data_loader.common.rec_text_process import RecTextProcess


class OCRDataSetProcess(Polygon2dDataSetProcess):

    def __init__(self, resize_type, normalize_type,
                 mean=0, std=1, pad_color=0, use_space=False):
        super().__init__(resize_type, normalize_type, mean, std, pad_color)
        self.text_process = RecTextProcess(use_space)

    def read_character(self, char_path):
        return self.text_process.read_character(char_path)

    def resize_dataset(self, src_image, image_size, ocr_objects):
        labels = []
        src_size = (src_image.shape[1], src_image.shape[0])  # [width, height]
        image = self.resize_image(src_image, image_size)
        for ocr in ocr_objects:
            temp_ocr = ocr.copy()
            points = ocr.get_polygon()
            polygon = self.resize_polygon(points, src_size, image_size)
            temp_ocr.clear_polygon()
            for point in polygon:
                temp_ocr.add_point(point)
            labels.append(temp_ocr)
        return image, labels

    def normalize_dataset(self, image, ocr_objects):
        text_polys = []
        texts = []
        image = self.dataset_process.normalize_image(image)
        for ocr in ocr_objects:
            points = self.get_four_points(ocr.get_polygon())
            text_polys.append(points)
            texts.append(ocr.get_text())
        result = {'texts': texts,
                  'text_polys': text_polys}
        return image, result


