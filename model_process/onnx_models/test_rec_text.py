#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from optparse import OptionParser
import numpy as np
import onnxruntime
from easyai.name_manager.task_name import TaskName
from easyai.config.utility.config_factory import ConfigFactory
from easyai.data_loader.utility.data_transforms_factory import DataTransformsFactory
from easyai.data_loader.common.text_data_loader import TextDataLoader
from easyai.tasks.rec_text.text_result_process import TextResultProcess


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program is onnx inference"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="image path txt")

    parser.add_option("-o", "--onnx", dest="onnx_path",
                      metavar="PATH", type="string", default=None,
                      help="onnx path")

    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.onnx_path:
        if not os.path.exists(options.onnx_path):
            parser.error("Could not find the onnx file")
        else:
            options.onnx_path = os.path.normpath(options.onnx_path)
    else:
        parser.error("'onnx' option is required to run this program")

    return options


class OnnxRecognizeTextTest():

    def __init__(self, onnx_path, config_path=None):
        self.config_factory = ConfigFactory()
        self.onnx_path = onnx_path
        self.session = onnxruntime.InferenceSession(self.onnx_path, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.task_config = self.config_factory.get_config(TaskName.RecognizeText,
                                                          config_path)
        self.transform_factory = DataTransformsFactory()
        self.result_process = TextResultProcess(self.task_config.character_set,
                                                self.task_config.post_process)
        self.image_size = tuple(self.task_config.data['image_size'])

    def test(self, val_path):
        dataloader = self.preprocess(val_path)
        for i, batch_data in enumerate(dataloader):
            image_data = batch_data['image']
            temp_data = image_data.squeeze(0)
            temp_data = temp_data.data.cpu().numpy()
            channels, height, width = temp_data.shape
            data = np.zeros((channels, height, self.image_size[0]), dtype=temp_data.dtype)
            data[:, :, 0:width] = temp_data
            data = np.expand_dims(data, axis=0)
            print(data.shape)
            prediction = self.session.run([self.output_name], {self.input_name: data})
            result = self.result_process.post_process(prediction[0])
            self.save_result(batch_data['file_path'], result)

    def preprocess(self, val_path):
        data_channel = self.task_config.data['data_channel']
        mean = self.task_config.data.get('mean', 1)
        std = self.task_config.data.get('std', 0)
        resize_type = self.task_config.data['resize_type']
        normalize_type = self.task_config.data['normalize_type']
        transform_args = self.task_config.data.get('transform_func', None)
        transform_func = self.transform_factory.get_data_transform(transform_args)
        dataloader = TextDataLoader(val_path, self.image_size, data_channel,
                                    resize_type, normalize_type, mean, std,
                                    transform_func)
        return dataloader

    def save_result(self, file_path, ocr_object):
        path, filename_post = os.path.split(file_path)
        with open(self.task_config.save_result_path, 'a') as file:
            file.write("{}|{} \n".format(filename_post, ocr_object[0].get_text()))

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def print_model_inputs(self):
        for x in self.session.get_inputs():
            print(x.name, x.shape)

    def print_model_outputs(self):
        for x in self.session.get_outputs():
            print(x.name, x.shape)


def main():
    print("process start...")
    options = parse_arguments()
    inference = OnnxRecognizeTextTest(options.onnx_path)
    inference.test(options.input_path)
    print("process end!")


if __name__ == "__main__":
    main()
