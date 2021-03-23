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
from easyai.helper.json_process import JsonProcess
from easyai.helper.arguments_parse import ToolArgumentsParse


class CreatePoseSample():

    def __init__(self,):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.annotation_name = "Annotations"
        self.images_dir_name = "JPEGImages"
        self.annotation_post = ".json"

    def create_train_and_test(self, input_dir, output_path, probability):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        annotations_dir = os.path.join(input_dir, "../%s" % self.annotation_name)
        save_train_path = os.path.join(output_path, "train.txt")
        save_val_path = os.path.join(output_path, "val.txt")
        if os.path.exists(save_train_path):
            print("%s exits" % save_train_path)
            return
        save_train_file_path = open(save_train_path, "w")
        save_test_file_path = open(save_val_path, "w")

        image_list = list(self.dir_process.getDirFiles(input_dir, "*.*"))
        random.shuffle(image_list)
        image_index = 0
        for image_path in image_list:
            # print(image_path)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(image_path)
            image_name, post = os.path.splitext(file_name_and_post)
            json_path = os.path.join(annotations_dir, "%s%s" % (image_name, self.annotation_post))
            _, boxes = self.json_process.parse_pose2d_data(json_path)
            if (image is not None) and len(boxes) > 0:
                image_index += 1
                if image_index % probability == 0:
                    save_test_file_path.write("%s\n" % file_name_and_post)
                else:
                    save_train_file_path.write("%s\n" % file_name_and_post)
        save_train_file_path.close()
        save_test_file_path.close()


def test():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreatePoseSample()
    if options.type.strip() == "train_val":
        test.create_train_and_test(options.inputPath,
                                   options.outputPath,
                                   options.probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()
