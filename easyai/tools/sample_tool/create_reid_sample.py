#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class CreateDetectionSample():

    def __init__(self, ):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.annotation_name = "Annotations"
        self.images_dir_name = "JPEGImages"
        self.annotation_post = ".json"

    def compute_max_id(self, input_dir):
        max_id = -1
        for dir_name in os.listdir(input_dir):
            image_dir_path = os.path.join(input_dir, dir_name, self.images_dir_name)
            annotations_dir_path = os.path.join(input_dir, dir_name, self.annotation_name)
            if not os.path.exists(image_dir_path) or not os.path.exists(annotations_dir_path):
                continue
            image_list = list(self.dir_process.getDirFiles(image_dir_path, "*.*"))
            for imageIndex, imagePath in enumerate(image_list):
                # print(imagePath)
                image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                path, file_name_and_post = os.path.split(imagePath)
                image_name, post = os.path.splitext(file_name_and_post)
                json_path = os.path.join(annotations_dir_path, "%s%s" % (image_name, self.annotation_post))
                if (image is not None) and os.path.exists(json_path):
                    _, temp_objects = self.json_process.parse_rect_data(json_path)
                    for temp in temp_objects:
                        if temp.objectId > max_id:
                            max_id = temp.objectId
        print("max_id:", max_id + 1)
        return max_id + 1

    def create_train_and_test(self, input_dir, output_path, probability):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_train_path = os.path.join(output_path, "train.txt")
        save_val_path = os.path.join(output_path, "val.txt")

        save_train_file_path = open(save_train_path, "w")
        save_test_file_path = open(save_val_path, "w")

        for video_index, dir_name in enumerate(os.listdir(input_dir)):
            image_dir_path = os.path.join(input_dir, dir_name, self.images_dir_name)
            annotations_dir_path = os.path.join(input_dir, dir_name, self.annotation_name)
            if not os.path.exists(image_dir_path) or not os.path.exists(annotations_dir_path):
                continue

            if (video_index + 1) % probability == 0:
                self.write_image_path(save_test_file_path, dir_name,
                                      image_dir_path, annotations_dir_path)
            else:
                self.write_image_path(save_train_file_path, dir_name,
                                      image_dir_path, annotations_dir_path)

        save_train_file_path.close()
        save_test_file_path.close()

    def write_image_path(self, save_file, dir_name, image_dir_path, annotations_dir_path):
        image_list = list(self.dir_process.getDirFiles(image_dir_path, "*.*"))
        for imageIndex, imagePath in enumerate(image_list):
            # print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            json_path = os.path.join(annotations_dir_path, "%s%s" % (image_name, self.annotation_post))
            if (image is not None) and os.path.exists(json_path):
                temp_save_str = dir_name + "/" + self.images_dir_name + "/" + file_name_and_post
                save_file.write("%s\n" % temp_save_str)

    def create_write_file(self, outputPath, class_name):
        result = {}
        for className in class_name:
            class_image_path = os.path.join(outputPath, className + ".txt")
            if not os.path.exists(class_image_path):
                result[className] = open(class_image_path, "w")
            else:
                print("%s exits" % class_image_path)
        return result


def test():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreateDetectionSample()
    if options.type.strip() == "train_val":
        test.create_train_and_test(options.inputPath,
                                   options.outputPath,
                                   options.probability)
    elif options.type.strip() == "max_id":
        test.compute_max_id(options.inputPath)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    test()




