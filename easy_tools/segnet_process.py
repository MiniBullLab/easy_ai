#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper.imageProcess import ImageProcess
from easyai.helper.dirProcess import DirProcess
from easyai.data_loader.utility.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.seg.segment_sample import SegmentSample


class SegNetProcess():

    def __init__(self):
        self.images_dir_name = "../JPEGImages"
        self.label_dir_name = "../SegmentLabel"
        self.annotation_post = ".png"
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()

    def resize_process(self, data_path):
        segment_sample = SegmentSample(data_path)
        segment_sample.read_sample()
        for img_path, label_path in segment_sample.image_and_label_list:
            src_image = self.image_process.opencvImageRead(img_path)
            label_image = self.image_process.opencvImageRead(label_path)
            if (src_image is not None) and (label_image is not None):
                src_size = src_image.shape[:2]
                label_size = label_image.shape[:2]
                if tuple(src_size) != tuple(label_size):
                    result = self.dataset_process.cv_image_resize(src_image,
                                                                  (label_size[1],
                                                                   label_size[0]))
                    self.image_process.opencv_save_image(label_path, result)
            else:
                print("(%s/%s) read segment data fail!" % (img_path, label_path))

    def png_process(self, data_path):
        temp_path, _ = os.path.split(data_path)
        labels_dir = os.path.join(temp_path, self.label_dir_name)
        for label_path in self.dir_process.getDirFiles(labels_dir, "*.*"):
            path, filename_and_post = os.path.split(label_path)
            filename, post = os.path.splitext(filename_and_post)
            if "png" not in post:
                image = self.image_process.opencvImageRead(label_path)
                label_filename = filename + self.annotation_post
                save_path = os.path.join(path, label_filename)
                if image is not None:
                    self.image_process.opencv_save_image(save_path, image)
                else:
                    print("%s read segment label fail!" % label_path)

