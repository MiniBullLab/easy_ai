#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code: test_create_detection_sample.py
# Author: wfw

import os
import sys
import json
sys.path.insert(0, os.getcwd() + "/..")

from easyai.tools.sample.create_detection_sample import CreateDetectionSample


def test_balance_sample(train_path, output_dir, class_name):
    print("start...")
    test = CreateDetectionSample()
    test.createBalanceSample(train_path,
                             output_dir,
                             class_name)
    print("End of game, have a nice day!")

def test_detection_sample(input_dir, output_path, probability):
    print("start...")
    test = CreateDetectionSample()
    test.createTrainAndTest(input_dir,
                            output_path,
                            probability)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    data_name = "Object365_small"
    class_file = open("/home/wfw/data/VOCdevkit/" + data_name + "/class.json", 'r')
    classes_ = json.load(class_file)
    classes = [value for key, value in classes_.items()]

    train_path = "/home/wfw/data/VOCdevkit/" + data_name + "/ImageSets/train_val.txt"
    output_dir = "/home/wfw/data/VOCdevkit/" + data_name + "/ImageSets"

    input_dir = "/home/wfw/data/VOCdevkit/" + data_name + "/JPEGImages"
    output_path = "/home/wfw/data/VOCdevkit/" + data_name + "/ImageSets"

    test_balance_sample(train_path, output_dir, classes)
    # test_detection_sample(input_dir, output_path, 10)
