#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import os
import shutil
from optparse import OptionParser
from easyai.helper import DirProcess


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program create test images"

    parser.add_option("-v", "--valPath", dest="valPath",
                      metavar="PATH", type="string", default="./val.txt",
                      help="path to data config file")

    (options, args) = parser.parse_args()

    # if options.valPath:
    #     if not os.path.exists(options.valPath):
    #         parser.error("Could not find the input file")
    #     else:
    #         options.valPath = os.path.normpath(options.valPath)
    # else:
    #     parser.error("'valPath' option is required to run this program")

    return options


class CreateTestImage():

    def __init__(self):
        self.save_dir = "./images"
        self.images_dir_name = "JPEGImages"
        self.dir_process = DirProcess()

    def create(self, val_path):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        temp_path, _ = os.path.split(val_path)
        images_dir = os.path.join(temp_path, "../%s" % self.images_dir_name)
        image_count = 0
        for line_data in self.dir_process.getFileData(val_path):
            data_list = [x.strip() for x in line_data.split() if x.strip()]
            image_path = os.path.join(images_dir, data_list[0])
            if os.path.isfile(image_path):
                path, image_name = os.path.split(image_path)
                new_path = os.path.join(self.save_dir, image_name)
                shutil.copy(image_path, new_path)
            else:
                data_list = [x.strip() for x in line_data.split("|") if x.strip()]
                image_path = os.path.join(images_dir, data_list[0])
                if os.path.isfile(image_path):
                    path, image_name = os.path.split(image_path)
                    new_path = os.path.join(self.save_dir, image_name)
                    shutil.copy(image_path, new_path)
            if image_count >= 10:
                break
            else:
                image_count += 1


if __name__ == "__main__":
    options = parse_arguments()
    test = CreateTestImage()
    test.create(options.valPath)

