#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
# import sys
# sys.path.insert(0, os.getcwd() + "/..")
import random
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper import ImageProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class CreateSegmentionSample():

    def __init__(self):
        self.dir_process = DirProcess()
        self.image_process = ImageProcess()
        self.images_dir_name = "../JPEGImages"
        self.segment_dir_name = "../SegmentLabel"
        self.annotation_post = ".png"

    def create_train_and_test(self, inputDir, outputPath, probability):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        annotations_dir = os.path.join(inputDir, self.segment_dir_name)
        save_train_path = os.path.join(outputPath, "train.txt")
        save_val_path = os.path.join(outputPath, "val.txt")
        if os.path.exists(save_train_path):
            data_result = self.read_data_text(save_train_path)
            if len(data_result) > 0:
                EasyLogger.debug("%s exits" % save_train_path)
                return
        save_train_file_path = open(save_train_path, "w")
        save_test_file_path = open(save_val_path, "w")

        image_list = list(self.dir_process.getDirFiles(inputDir, "*.*"))
        random.shuffle(image_list)
        for imageIndex, imagePath in enumerate(image_list):
            # print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            seg_label_name = image_name + self.annotation_post
            label_path = os.path.join(annotations_dir, seg_label_name)
            if (image is not None) and os.path.exists(label_path):
                if (imageIndex + 1) % probability == 0:
                    save_test_file_path.write("%s\n" % file_name_and_post)
                else:
                    save_train_file_path.write("%s\n" % file_name_and_post)
        save_train_file_path.close()
        save_test_file_path.close()

    def read_data_text(self, data_path):
        result = []
        temp_path, _ = os.path.split(data_path)
        images_dir = os.path.join(temp_path, self.images_dir_name)
        annotations_dir = os.path.join(temp_path, self.segment_dir_name)
        for line_data in self.dir_process.getFileData(data_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            image_path = os.path.join(images_dir, data_list[0])
            image_name, post = os.path.splitext(data_list[0])
            seg_label_name = image_name + self.annotation_post
            label_path = os.path.join(annotations_dir, seg_label_name)
            if os.path.exists(image_path) and os.path.exists(label_path):
                result.append(image_path)
        return result


def test():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreateSegmentionSample()
    test.create_train_and_test(options.inputPath,
                               options.outputPath,
                               options.probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()



