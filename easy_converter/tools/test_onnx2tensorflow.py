#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_converter.converter.onnx_convert_tensorflow import OnnxConvertTensorflow
from easy_converter.helper.arguments_parse import TaskArgumentsParse


def main():
    print("process start...")
    options = TaskArgumentsParse.onnx2tensorflow_parse_arguments()
    converter = OnnxConvertTensorflow(options.input_path)
    converter.convert_tensorflow()
    print("process end!")


if __name__ == "__main__":
    main()