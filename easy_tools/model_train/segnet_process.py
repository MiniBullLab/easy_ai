#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
from easyai.helper.image_process import ImageProcess
from easyai.helper.dir_process import DirProcess
from easyai.data_loader.common.image_dataset_process import ImageDataSetProcess
from easyai.data_loader.seg.segment_sample import SegmentSample
from easyai.tools.sample_tool.sample_info_get import SampleInformation
from easyai.tools.sample_tool.convert_segment_label import ConvertSegmentionLable
from easyai.name_manager.task_name import TaskName
from easyai.utility.logger import EasyLogger


class SegNetProcess():

    def __init__(self):
        self.images_dir_name = "../JPEGImages"
        self.annotation_dir_name = "../Annotations"
        self.annotation_post = ".json"
        self.save_label_dir = "../SegmentLabel"
        self.segment_post = ".png"
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.dataset_process = ImageDataSetProcess()
        self.convert_label = ConvertSegmentionLable()
        self.sample_process = SampleInformation()

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

    def label_process(self, data_path):
        temp_path, _ = os.path.split(data_path)
        labels_dir = os.path.join(temp_path, self.save_label_dir)
        annotation_dir = os.path.join(temp_path, self.annotation_dir_name)
        if not os.path.exists(labels_dir):
            if os.path.exists(annotation_dir):
                class_names = self.sample_process.create_class_names(data_path,
                                                                     TaskName.Detect2d_Task)
                if class_names is not None and len(class_names) > 0:
                    self.convert_label.convert_segment_label(temp_path, -2, class_names)
                else:
                    EasyLogger.error("input segnet datset error!")
            else:
                EasyLogger.error("input segnet datset error!")
        else:
            self.png_process(data_path)

    def png_process(self, data_path):
        temp_path, _ = os.path.split(data_path)
        labels_dir = os.path.join(temp_path, self.save_label_dir)
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


