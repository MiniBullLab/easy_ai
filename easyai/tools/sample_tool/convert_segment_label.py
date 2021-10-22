#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("tools.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")
import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import cv2
import numpy as np
from easyai.helper import DirProcess
from easyai.helper import ImageProcess
from easyai.helper.json_process import JsonProcess
from easyai.helper.arguments_parse import ToolArgumentsParse
from easyai.config.utility.config_factory import ConfigFactory
from easyai.name_manager.task_name import TaskName


class ConvertSegmentionLable():

    def __init__(self):
        self.images_dir_name = "../JPEGImages"
        self.annotation_dir_name = "../Annotations"
        self.annotation_post = ".json"
        self.save_label_dir = "../SegmentLabel"
        self.segment_post = ".png"
        self.label_pad_color = 250
        self.volid_label_seg = []
        self.valid_label_seg = [[0], [1]]
        self.dirProcess = DirProcess()
        self.json_process = JsonProcess()
        self.image_process = ImageProcess()

    def convert_segment_label(self, label_dir, label_type, class_list):
        if label_type > 0:
            output_dir = os.path.join(label_dir, self.save_label_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for label_path in self.dirProcess.getDirFiles(label_dir, "*.*"):
                _, file_name_and_post = os.path.split(label_path)
                EasyLogger.debug(label_path)
                mask = self.process_segment_label(label_path, label_type, class_list)
                if mask is not None:
                    save_path = os.path.join(output_dir, file_name_and_post)
                    cv2.imwrite(save_path, mask)
        elif label_type < 0:
            output_dir = os.path.join(label_dir, self.save_label_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            annotation_dir = os.path.join(label_dir, self.annotation_dir_name)
            for label_path in self.dirProcess.getDirFiles(annotation_dir, "*.json"):
                _, file_name_post = os.path.split(label_path)
                file_name, post = os.path.splitext(file_name_post)
                EasyLogger.debug(label_path)
                mask = self.convert_json_to_segment_label(label_path, label_type, class_list)
                if mask is not None:
                    save_name_and_post = file_name + self.segment_post
                    save_path = os.path.join(output_dir, save_name_and_post)
                    cv2.imwrite(save_path, mask)

    def process_segment_label(self, label_path, label_type, class_list):
        mask = None
        if label_type == 1:  # gray
            mask = self.image_process.read_gray_image(label_path)
        elif label_type == 2:  # rgb
            _, mask = self.image_process.readRgbImage(label_path)
        if mask is not None:
            if label_type == 1:  # gray
                mask = self.convert_gray_label(mask, class_list)
            elif label_type == 2:  # rgb
                mask = self.convert_color_label(mask, class_list)
        return mask

    def convert_json_to_segment_label(self, json_path, label_type, class_list):
        mask = None
        file_name_post, polygon_list = self.json_process.parse_segment_data(json_path)
        path, _ = os.path.split(json_path)
        image_dir = os.path.join(path, self.images_dir_name)
        image_path = os.path.join(image_dir, file_name_post)
        if os.path.exists(image_path):
            image_data = self.image_process.opencvImageRead(image_path)
            if image_data is None:
                return mask
        else:
            return mask
        if label_type == -1:  # gray
            mask = self.fill_gray_label(image_data.shape[:2], polygon_list, class_list)
        elif label_type == -2:  # rgb
            mask = self.fill_color_label(image_data.shape[:2], polygon_list, class_list)
        return mask

    def merge_segment_label(self, mask, volid_label, valid_label):
        classes = -np.ones([100, 100])
        valid = [x for j in valid_label for x in j]
        for i in range(0, len(valid_label)):
            classes[i, :len(valid_label[i])] = valid_label[i]
        for label in volid_label:
            mask[mask == label] = self.label_pad_color
        for validc in valid:
            mask[mask == validc] = np.uint8(np.where(classes == validc)[0])
        return mask

    def convert_gray_label(self, mask, class_list):
        shape = mask.shape  # shape = [height, width]
        result = np.full(shape, 250, dtype=np.uint8)
        for index, value in enumerate(class_list):
            gray_value = int(value[1].strip())
            result[mask == gray_value] = index
        return result

    def convert_color_label(self, mask, class_list):
        shape = mask.shape[:2]  # shape = [height, width]
        result = np.full(shape, 250, dtype=np.uint8)
        for index, value in enumerate(class_list):
            value_list = [int(x) for x in value[1].split(',') if x.strip()]
            color_value = np.array(value_list, dtype=np.uint8)
            temp1 = mask[:, :] == color_value
            temp2 = np.sum(temp1, axis=2)
            result[temp2 == 3] = index
        return result

    def fill_gray_label(self, shape, polygon_list, class_list):
        result = np.full(shape, 255, dtype=np.uint8)
        for class_name, value in class_list:
            if class_name.strip() == "background":
                continue
            for polygon_object in polygon_list:
                if class_name.strip() == polygon_object.name:
                    contours = np.array([[p.x, p.y] for p in polygon_object.get_polygon()])
                    cv2.fillPoly(result, [contours], value)
        return result

    def fill_color_label(self, shape, polygon_list, class_list):
        result = np.full(shape, (255, 255, 255), dtype=np.uint8)
        for class_name, value in class_list:
            if class_name.strip() == "background":
                continue
            for polygon_object in polygon_list:
                if class_name.strip() == polygon_object.name:
                    contours = np.array([[p.x, p.y] for p in polygon_object.get_polygon()])
                    cv2.fillPoly(result, [contours], value)
        return result


def main():
    print("start...")
    options = ToolArgumentsParse.dir_path_parse()
    test = ConvertSegmentionLable()
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(TaskName.Segment_Task, config_path=options.config_path)
    test.convert_segment_label(options.inputPath,
                               task_config.seg_label_type,
                               task_config.segment_class)
    print("End of game, have a nice day!")


if __name__ == "__main__":
    main()
