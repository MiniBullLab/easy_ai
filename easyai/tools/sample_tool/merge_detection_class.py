#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import json
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper.json_process import JsonProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.utility.logger import EasyLogger


class MergeDetectionClass():

    def __init__(self, ):
        self.dir_process = DirProcess()
        self.json_process = JsonProcess()
        self.annotation_name = "../Annotations"
        self.images_dir_name = "../JPEGImages"
        self.annotation_post = "*.json"

    def merge_class(self, input_dir, output_path, merge_dict):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image_dir = os.path.join(input_dir, self.images_dir_name)
        for label_path in self.dir_process.getDirFiles(input_dir, self.annotation_post):
            print(label_path)
            _, file_name_and_post = os.path.split(label_path)
            image_name, boxes = self.json_process.parse_rect_data(label_path)
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                 cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            result = []
            for box in boxes:
                for key, value in merge_dict.items():
                    if box.name.strip() in value:
                        box.name = key
                        result.append(box)
                        break
            if len(result) == 0:
                continue
            save_path = os.path.join(output_path, file_name_and_post)
            image_size = (image.shape[1], image.shape[0])  # [width, height]
            self.json_write("merge_dataset", image_name, image_path, image_size, result, save_path)

    def json_write(self, database, file_name, file_path, image_size, boxes, json_path):
        annotation = dict()
        # annotation
        annotation['annotation'] = 'Annotations'
        # database
        annotation['database'] = database
        # owner
        annotation['owner'] = 'miniBull'
        # folder
        annotation['folder'] = 'JPEGImages'
        # filename
        annotation['filename'] = file_name
        # path
        annotation['path'] = file_path
        # size
        annotation['size'] = {'width': image_size[0], 'height': image_size[1], 'depth': 3}
        # objectCount
        annotation['objectCount'] = len(boxes)
        # objects
        rectObject = []
        for box in boxes:
            rectObject.append({'class': box.name,
                               'minX': box.min_corner.x,
                               'minY': box.min_corner.y,
                               'maxX': box.max_corner.x,
                               'maxY': box.max_corner.y})
        annotation['objects'] = {'rectObject': rectObject}

        a = json.dumps(annotation, indent=4)
        f = open(json_path, 'w')
        f.write(a)
        f.close()


def main():
    print("start...")
    test = MergeDetectionClass()
    test.merge_class("/home/lpj/dataset/det2d/Berkeley_dataset/Annotations",
                     "/home/lpj/dataset/det2d/Berkeley_dataset/merge_Annotations",
                     {"car": ['car', 'bus', 'truck']})
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()




