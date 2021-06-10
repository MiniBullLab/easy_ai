#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")

import random
import numpy as np
import cv2
from easyai.helper.dir_process import DirProcess
from easyai.helper.arguments_parse import ToolArgumentsParse


class CreateOneClassSample():

    def __init__(self):
        self.dir_process = DirProcess()

    def process_sample(self, input_dir, output_dir, flag, probability=1):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if "train_val" == flag.strip():
            self.process_train_val(input_dir, output_dir, probability)
        elif "train" == flag.strip():
            self.process_train(input_dir, output_dir, flag)
        elif "val" == flag.strip():
            self.process_val(input_dir, output_dir, flag)

    def process_train_val(self, input_dir, output_dir, probability):
        intput_path, ok_dir_name = os.path.split(input_dir)
        data_class = self.get_data_class(intput_path)
        print("ok dir:", ok_dir_name)
        assert len(data_class) == 2 or len(data_class) == 1

        save_train_path = os.path.join(output_dir, "train.txt")
        save_val_path = os.path.join(output_dir, "val.txt")
        if os.path.exists(save_train_path):
            print("%s exits" % save_train_path)
            return
        save_train_file = open(save_train_path, "w")
        save_val_file = open(save_val_path, "w")

        for class_index, class_name in enumerate(data_class):
            data_class_dir = os.path.join(intput_path, class_name)
            image_list = list(self.dir_process.getDirFiles(data_class_dir, "*.*"))
            random.shuffle(image_list)
            if class_name == ok_dir_name:
                for image_index, image_path in enumerate(image_list):
                    # print(image_path)
                    if (image_index + 1) % probability == 0:
                        self.write_data(image_path, class_name, 0, save_val_file)
                    else:
                        self.write_data(image_path, class_name, 0, save_train_file)
            else:
                for image_index, image_path in enumerate(image_list):
                    # print(image_path)
                    self.write_data(image_path, class_name, 1, save_val_file)

        save_train_file.close()
        save_val_file.close()

    def process_train(self, input_dir, output_dir, flag):
        intput_path, ok_dir_name = os.path.split(input_dir)
        save_train_path = os.path.join(output_dir, "%s.txt" % flag)
        if os.path.exists(save_train_path):
            return
        save_train_file = open(save_train_path, "w")

        image_list = list(self.dir_process.getDirFiles(input_dir, "*.*"))
        random.shuffle(image_list)
        for image_index, image_path in enumerate(image_list):
            # print(image_path)
            self.write_data(image_path, ok_dir_name, 0, save_train_file)
        save_train_file.close()

    def process_val(self, input_dir, output_dir, flag):
        intput_path, ok_dir_name = os.path.split(input_dir)
        data_class = self.get_data_class(intput_path)
        print("ok dir:", ok_dir_name)
        assert len(data_class) == 2 or len(data_class) == 1
        save_val_path = os.path.join(output_dir, "%s.txt" % flag)
        if os.path.exists(save_val_path):
            return
        save_val_file = open(save_val_path, "w")

        for class_index, class_name in enumerate(data_class):
            data_class_dir = os.path.join(intput_path, class_name)
            image_list = list(self.dir_process.getDirFiles(data_class_dir, "*.*"))
            random.shuffle(image_list)
            if class_name == ok_dir_name:
                for image_index, image_path in enumerate(image_list):
                    # print(image_path)
                    self.write_data(image_path, class_name, 0, save_val_file)
            else:
                for image_index, image_path in enumerate(image_list):
                    # print(image_path)
                    self.write_data(image_path, class_name, 1, save_val_file)

        save_val_file.close()

    def write_data(self, image_path, class_name, class_index, save_file):
        path, fileNameAndPost = os.path.split(image_path)
        fileName, post = os.path.splitext(fileNameAndPost)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        if image is not None:
            write_content = "%s/%s %d\n" % (class_name, fileNameAndPost,
                                            class_index)
            save_file.write(write_content)

    def get_data_class(self, data_dir):
        result = []
        dir_names = os.listdir(data_dir)
        for name in dir_names:
            if not name.startswith("."):
                file_path = os.path.join(data_dir, name)
                if os.path.isdir(file_path):
                    result.append(name)
        return sorted(result)


def main():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreateOneClassSample()
    test.process_sample(options.inputPath,
                        options.outputPath,
                        options.type,
                        options.probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
