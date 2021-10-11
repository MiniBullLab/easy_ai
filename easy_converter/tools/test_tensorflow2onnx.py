#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_converter.converter.tensorflow_convert_onnx import TensorflowConvertOnnx
from easy_converter.helper.arguments_parse import TaskArgumentsParse


def main():
    print("process start...")
    options = TaskArgumentsParse.tensorflow2onnx_parse_arguments()
    converter = TensorflowConvertOnnx(options.pb_path)
    converter.convert_onnx()
    print("process end!")


if __name__ == "__main__":
    main()
