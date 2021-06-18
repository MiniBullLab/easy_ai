#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_converter.converter.keras_convert_onnx import KerasConvertOnnx
from easy_converter.helper.arguments_parse import TaskArgumentsParse


def main():
    print("process start...")
    options = TaskArgumentsParse.keras2onnx_parse_arguments()
    converter = KerasConvertOnnx(options.h5_path)
    converter.convert_onnx_from_h5(options.model_name)
    print("process end!")


if __name__ == "__main__":
    main()