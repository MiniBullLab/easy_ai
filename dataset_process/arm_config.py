#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import codecs
import json
import random
from optparse import OptionParser
from easyai.name_manager.task_name import TaskName
from easyai.tools.sample_tool.sample_info_get import SampleInformation


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program config create"

    parser.add_option("-t", "--task", dest="task_name",
                      type="string", default=None,
                      help="task name")

    parser.add_option("-i", "--trainPath", dest="trainPath",
                      metavar="PATH", type="string", default="./train.txt",
                      help="path to data config file")

    (options, args) = parser.parse_args()

    return options


class ARMConfig():

    def __init__(self):
        self.sample_process = SampleInformation()

    def create_config(self, task_name, train_path):
        if task_name.strip() == "ClassNET":
            self.create_classnet_config(train_path)
        elif task_name.strip() == "DeNET":
            self.create_denet_config(train_path)
        elif task_name.strip() == "SegNET":
            self.create_segnet_config(train_path)
        else:
            print("input task error!")

    def create_classnet_config(self, train_path):
        class_names = self.sample_process.create_class_names(train_path,
                                                             TaskName.Classify_Task)

        save_path = "classnet.json"
        save_data = dict()
        save_data['input_layer'] = "0"
        save_data['output_layer'] = "192"
        save_data['objects_name'] = class_names
        save_data['threshold'] = 0.3
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    def create_denet_config(self, train_path):
        output_name = ["636", "662", "688"]
        class_names = self.sample_process.create_class_names(train_path,
                                                             TaskName.Detect2d_Task)

        save_path = "denet.json"
        anchors = [9, 8.57, 12.43, 26.71, 19.71, 14.43, 26.36,
                   58.52, 36, 25.55, 64.42, 42.9, 96.44, 79, 158,
                   115, 218.65, 192.9]
        save_data = dict()
        for index, temp_name in enumerate(output_name, 1):
            output_layer = "output_layer_%d" % index
            save_data[output_layer] = temp_name
        save_data['objects_name'] = class_names
        for temp_name in class_names:
            save_data[temp_name] = [random.randint(0, 255),
                                    random.randint(0, 255),
                                    random.randint(0, 255)]
        save_data['image_width'] = 416
        save_data['image_height'] = 416
        save_data['threshold'] = 0.1
        save_data['anchors'] = anchors
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    def create_segnet_config(self, train_path):
        save_path = "segnet.json"
        save_data = dict()
        save_data['input_layer'] = "0"
        save_data['output_layer'] = "507"
        save_data['image_width'] = 504
        save_data['image_height'] = 400
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    options = parse_arguments()
    test = ARMConfig()
    test.create_config(options.task_name, options.trainPath)
