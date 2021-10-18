#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: lipeijie

import codecs
import json
import random


class ARMConfig():

    def __init__(self):
        pass

    def create_classnet_config(self, input_name, output_name,
                               objects_name, threshold=0.3):
        save_path = "classnet.json"
        save_data = dict()
        save_data['input_layer'] = input_name[0]
        save_data['output_layer'] = output_name[0]
        save_data['objects_name'] = objects_name
        save_data['threshold'] = threshold
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    def create_denet_config(self, input_name, output_name,
                            objects_name, image_width=416, image_height=416,
                            threshold=0.1):
        save_path = "denet.json"
        anchors = [10, 10, 24, 26, 33, 68,
                   61, 131, 70, 36, 117, 78,
                   128, 209, 277, 107, 311, 251]
        save_data = dict()
        for index, temp_name in enumerate(output_name, 1):
            output_layer = "output_layer_%d" % index
            save_data[output_layer] = temp_name
        save_data['objects_name'] = objects_name
        objects_color = dict()
        for temp_name in objects_name:
            objects_color[temp_name] = [random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255)]
        save_data['image_width'] = image_width
        save_data['image_height'] = image_height
        save_data['threshold'] = threshold
        save_data['anchors'] = anchors
        save_data['objects_color'] = objects_color
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    def create_segnet_config(self, input_name, output_name,
                             image_width=504, image_height=400):
        save_path = "segnet.json"
        save_data = dict()
        save_data['input_layer'] = input_name[0]
        save_data['output_layer'] = output_name[0]
        save_data['image_width'] = image_width
        save_data['image_height'] = image_height
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)

    def create_textnet_config(self, input_name, output_name, threshold=0.1):
        save_path = "textnet.json"
        save_data = dict()
        save_data['input_layer'] = input_name[0]
        save_data['output_layer'] = output_name[0]
        save_data['threshold'] = threshold
        with codecs.open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, sort_keys=True, indent=4, ensure_ascii=False)
