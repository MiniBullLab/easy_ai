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
from easyai.data_loader.common.rec_text_process import RecTextProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class CreateRecognizeTextSample():

    def __init__(self, ):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.image_process = ImageProcess()
        self.dataset_process = Polygon2dDataSetProcess(0, 0, 0, 0, 0)
        self.text_process = RecTextProcess(True)
        self.annotation_name = "../Annotations"
        self.images_dir_name = "../JPEGImages"
        self.annotation_post = ".json"
        self.image_save_dir = "rec_text"
        self.expand_ratio = (1.0, 1.0)

    def create_train_and_test(self, input_dir, output_path, probability,
                              language=("english",), char_path=None):
        if len(language) > 0:
            self.image_save_dir = ("".join(language)).strip()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        annotations_dir = os.path.join(input_dir, self.annotation_name)
        save_train_path = os.path.join(output_path, "train.txt")
        save_val_path = os.path.join(output_path, "val.txt")
        if os.path.exists(save_train_path):
            data_result = self.read_data_text(save_train_path)
            EasyLogger.debug("image count: %d" % len(data_result))
            if len(data_result) > 0:
                EasyLogger.info("%s exits" % save_train_path)
                return
        else:
            EasyLogger.debug("%s not exits" % save_train_path)
        save_train_file = open(save_train_path, "w", encoding='utf-8')
        save_test_file = open(save_val_path, "w", encoding='utf-8')

        if char_path is not None:
            char_list = self.text_process.read_character(char_path)
        else:
            char_list = []

        image_list = list(self.dir_process.getDirFiles(input_dir, "*.*"))
        random.shuffle(image_list)
        sample_index = 0
        for image_index, image_path in enumerate(image_list):
            # print(image_path)
            src_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            path, file_name_and_post = os.path.split(image_path)
            image_name, post = os.path.splitext(file_name_and_post)
            json_path = os.path.join(annotations_dir, "%s%s" % (image_name, self.annotation_post))
            if (src_image is not None) and os.path.exists(json_path):
                _, ocr_objects = self.json_process.parse_ocr_data(json_path)
                for ocr in ocr_objects:
                    text_data = ocr.get_text()
                    if len(char_list) > 0 and \
                            True in [c not in char_list for c in text_data.strip()]:
                        continue
                    if ocr.language.strip() in language:
                        image = self.dataset_process.get_rotate_crop_image(src_image,
                                                                           ocr.get_polygon()[:],
                                                                           self.expand_ratio)
                        dst_img_height, dst_img_width = image.shape[0:2]
                        if dst_img_height < 10 or dst_img_width < 10:
                            EasyLogger.warn("%s: small eara(%s)" % (image_path,
                                                                    ocr.get_text()))
                            continue
                        if dst_img_height * 1.0 / dst_img_width >= 2:
                            image = np.rot90(image)
                            EasyLogger.warn("%s: %.3f|%s(rotate 90)" % (image_path,
                                                                        dst_img_height * 1.0 / dst_img_width,
                                                                        ocr.get_text()))
                        if (sample_index + 1) % probability == 0:
                            self.write_data(image_path, image, ocr,
                                            sample_index, save_test_file)
                        else:
                            self.write_data(image_path, image, ocr,
                                            sample_index, save_train_file)
                        sample_index += 1
        save_train_file.close()
        save_test_file.close()

    def write_data(self, image_path, image, ocr, sample_index, save_file):
        path, file_name_post = os.path.split(image_path)
        file_name, post = os.path.splitext(file_name_post)
        save_name = file_name + "_%08d" % sample_index + post
        save_dir = os.path.join(path, self.image_save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        if image is not None:
            self.image_process.opencv_save_image(save_path, image)
            write_content = "%s/%s|%s\n" % (self.image_save_dir, save_name,
                                            ocr.get_text())
            save_file.write(write_content)

    def read_data_text(self, data_path, separator="|"):
        result = []
        temp_path, _ = os.path.split(data_path)
        images_dir = os.path.join(temp_path, self.images_dir_name)
        for line_data in self.dir_process.getFileData(data_path):
            data_list = [x.strip() for x in line_data.split(separator) if x.strip()]
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
                               ("english",), "./easyai/config/character/temp_en.txt")
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
