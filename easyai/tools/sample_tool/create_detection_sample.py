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
from easyai.tools.sample_tool.sample_info_get import SampleInformation
from easyai.name_manager.task_name import TaskName
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class CreateDetectionSample():

    def __init__(self,):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.annotation_name = "../Annotations"
        self.images_dir_name = "../JPEGImages"
        self.annotation_post = ".json"

    def create_balance_sample(self, inputTrainPath, outputPath, class_names):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        path, _ = os.path.split(inputTrainPath)
        annotationDir = os.path.join(path, self.annotation_name)
        imagesDir = os.path.join(path, self.images_dir_name)
        writeFile = self.create_write_file(outputPath, class_names)
        if len(writeFile) == 0:
            return
        for fileNameAndPost in self.dir_process.getFileData(inputTrainPath):
            fileName, post = os.path.splitext(fileNameAndPost)
            annotationFileName = fileName + self.annotation_post
            annotationPath = os.path.join(annotationDir, annotationFileName)
            imagePath = os.path.join(imagesDir, fileNameAndPost)
            print(imagePath, annotationPath)
            if os.path.exists(annotationPath) and \
               os.path.exists(imagePath):
                _, boxes = self.json_process.parse_rect_data(annotationPath)
                allNames = [box.name for box in boxes if box.name in class_names]
                names = set(allNames)
                print(names)
                for className in names:
                    writeFile[className].write(fileNameAndPost + "\n")

    def create_train_and_test(self, input_dir, output_path, probability):
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
        save_train_file_path = open(save_train_path, "w")
        save_test_file_path = open(save_val_path, "w")

        imageList = list(self.dir_process.getDirFiles(input_dir, "*.*"))
        random.shuffle(imageList)
        for imageIndex, imagePath in enumerate(imageList):
            # print(imagePath)
            image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            path, file_name_and_post = os.path.split(imagePath)
            image_name, post = os.path.splitext(file_name_and_post)
            json_path = os.path.join(annotations_dir, "%s%s" % (image_name, self.annotation_post))
            if (image is not None) and os.path.exists(json_path):
                if (imageIndex + 1) % probability == 0:
                    save_test_file_path.write("%s\n" % file_name_and_post)
                else:
                    save_train_file_path.write("%s\n" % file_name_and_post)
        save_train_file_path.close()
        save_test_file_path.close()

    def create_write_file(self, outputPath, class_name):
        result = {}
        for className in class_name:
            class_image_path = os.path.join(outputPath, className + ".txt")
            if not os.path.exists(class_image_path):
                result[className] = open(class_image_path, "w")
            else:
                print("%s exits" % class_image_path)
        return result
    
    def read_data_text(self, data_path):
        result = []
        temp_path, _ = os.path.split(data_path)
        images_dir = os.path.join(temp_path, self.images_dir_name)
        annotations_dir = os.path.join(temp_path, self.annotation_name)
        for line_data in self.dir_process.getFileData(data_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            image_path = os.path.join(images_dir, data_list[0])
            image_name, post = os.path.splitext(data_list[0])
            json_path = os.path.join(annotations_dir,
                                     "%s%s" % (image_name, self.annotation_post))
            if os.path.exists(image_path) and os.path.exists(json_path):
                result.append(image_path)
        return result


def test():
    print("start...")
    options = ToolArgumentsParse.process_sample_parse()
    test = CreateDetectionSample()
    if options.type.strip() == "train_val":
        test.create_train_and_test(options.inputPath,
                                   options.outputPath,
                                   options.probability)
    elif options.type.strip() == "balance":
        sample_process = SampleInformation()
        class_names = sample_process.create_class_names(options.inputPath,
                                                        TaskName.Detect2d_Task)
        test.create_balance_sample(options.inputPath,
                                   options.outputPath,
                                   class_names)
    print("End of game, have a nice day!")


if __name__ == "__main__":
   test()




