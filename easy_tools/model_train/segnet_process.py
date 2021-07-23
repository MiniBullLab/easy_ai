#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper.image_process import ImageProcess
from easyai.helper.dir_process import DirProcess
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.utility.logger import EasyLogger


class SegNetProcess():

    def __init__(self):
        self.label_dir_name = "SegmentLabel"
        self.annotation_post = ".png"
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()

    def resize_process(self, data_path):
        image_count = 0
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
                    self.image_process.opencv_save_image(img_path, result)
                image_count += 1
            else:
                EasyLogger.error("(%s/%s) read segment data fail!" % (img_path, label_path))
        # assert image_count > 0

    def png_process(self, data_path):
        temp_path, _ = os.path.split(data_path)
        root_path, _ = os.path.split(temp_path)
        labels_dir = os.path.join(root_path, self.label_dir_name)
        for label_path in self.dir_process.getDirFiles(labels_dir, "*.*"):
            path, filename_and_post = os.path.split(label_path)
            filename, post = os.path.splitext(filename_and_post)
            if "png" not in post:
                image = self.image_process.opencvImageRead(label_path)
                label_filename = filename + self.annotation_post
                save_path = os.path.join(path, label_filename)
                if image is not None:
                    self.image_process.opencv_save_image(save_path, image)
                    os.remove(label_path)
                else:
                    EasyLogger.error("%s read segment label fail!" % label_path)

        def label_convert():
            pass

