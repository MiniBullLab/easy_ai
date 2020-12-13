#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import random
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper import ImageProcess
from easyai.helper.arguments_parse import ToolArgumentsParse


class CreateSegmentionSample():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.image_process = ImageProcess()
        self.segment_dir_name = "SegmentLabel"
        self.annotation_post = ".png"

    def create_train_and_test(self, inputDir, outputPath, probability):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        annotations_dir = os.path.join(inputDir, "../%s" % self.segment_dir_name)
        save_train_path = os.path.join(outputPath, "train.txt")
        save_val_path = os.path.join(outputPath, "val.txt")
        if os.path.exists(save_train_path):
            print("%s exits" % save_train_path)
            return
        save_train_file_path = open(save_train_path, "w")
        save_test_file_path = open(save_val_path, "w")

        imageList = list(self.dirProcess.getDirFiles(inputDir, "*.*"))
        random.shuffle(imageList)
        for imageIndex, imagePath in enumerate(imageList):
            # print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            seg_label_name = image_name + self.annotation_post
            label_path = os.path.join(annotations_dir, seg_label_name)
            if (image is not None) and os.path.exists(label_path):
                if (imageIndex + 1) % probability == 0:
                    save_train_file_path.write("%s\n" % file_name_and_post)
                else:
                    save_test_file_path.write("%s\n" % file_name_and_post)
        save_train_file_path.close()
        save_test_file_path.close()


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


