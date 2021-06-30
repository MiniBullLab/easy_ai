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
from easyai.helper.json_process import JsonProcess
from easyai.helper.image_process import ImageProcess
from easyai.data_loader.common.polygon2d_dataset_process import Polygon2dDataSetProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class CreateRecognizeTextSample():

    def __init__(self, ):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.image_process = ImageProcess()
        self.dataset_process = Polygon2dDataSetProcess(0, 0, 0, 0, 0)
        self.annotation_name = "../Annotations"
        self.images_dir_name = "../JPEGImages"
        self.annotation_post = ".json"
        self.image_save_dir = "rec_text"
        self.expand_ratio = (1.0, 1.0)

    def create_train_and_test(self, input_dir, output_path, probability,
                              language=("english",)):
        self.image_save_dir = ("".join(language)).strip()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        annotations_dir = os.path.join(input_dir, self.annotation_name)
        save_train_path = os.path.join(output_path, "train.txt")
        save_val_path = os.path.join(output_path, "val.txt")
        if os.path.exists(save_train_path):
            data_result = self.read_data_text(save_train_path)
            if len(data_result) > 0:
                EasyLogger.debug("%s exits" % save_train_path)
                return
        save_train_file_path = open(save_train_path, "w", encoding='utf-8')
        save_test_file_path = open(save_val_path, "w", encoding='utf-8')

        image_list = list(self.dir_process.getDirFiles(input_dir, "*.*"))
        random.shuffle(image_list)
        sample_index = 0
        for image_index, imagePath in enumerate(image_list):
            # print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_COLOR)
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            json_path = os.path.join(annotations_dir, "%s%s" % (image_name, self.annotation_post))
            if (image is not None) and os.path.exists(json_path):
                _, ocr_objects = self.json_process.parse_ocr_data(json_path)
                for ocr in ocr_objects:
                    if ocr.language.strip() in language:
                        if (sample_index + 1) % probability == 0:
                            self.write_data(imagePath, image, ocr,
                                            sample_index, save_test_file_path)
                        else:
                            self.write_data(imagePath, image, ocr,
                                            sample_index, save_train_file_path)
                        sample_index += 1
        save_train_file_path.close()
        save_test_file_path.close()

    def write_data(self, image_path, src_image, ocr, sample_index, save_file):
        path, file_name_post = os.path.split(image_path)
        file_name, post = os.path.splitext(file_name_post)
        save_name = file_name + "_%08d" % sample_index + post
        save_dir = os.path.join(path, self.image_save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        image = self.dataset_process.get_rotate_crop_image(src_image,
                                                           ocr.get_polygon()[:],
                                                           self.expand_ratio)
        if image is not None:
            self.image_process.opencv_save_image(save_path, image)
            write_content = "%s/%s|%s\n" % (self.image_save_dir, save_name,
                                            ocr.get_text())
            save_file.write(write_content)

    def read_data_text(self, data_path):
        result = []
        temp_path, _ = os.path.split(data_path)
        images_dir = os.path.join(temp_path, self.images_dir_name)
        for line_data in self.dir_process.getFileData(data_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            image_path = os.path.join(images_dir, data_list[0])
            if os.path.exists(image_path):
                result.append(image_path)
        return result


def main():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreateRecognizeTextSample()
    test.create_train_and_test(options.inputPath,
                               options.outputPath,
                               options.probability,
                               ("english",))
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
