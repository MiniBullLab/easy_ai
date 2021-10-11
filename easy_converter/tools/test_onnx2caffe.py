#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easy_converter.converter.onnx_convert_caffe import OnnxConvertCaffe
from easy_converter.helper.arguments_parse import TaskArgumentsParse


def main():
    print("process start...")
    options = TaskArgumentsParse.onnx2caffe_parse_arguments()
    converter = OnnxConvertCaffe(options.input_path)
    if options.proto_path is None:
        converter.convert_caffe()
    elif os.path.exists(options.proto_path):
        converter.convert_weights(options.proto_path)
    else:
        print("input error")
    print("process end!")


if __name__ == "__main__":
    main()
