#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_converter.converter.darknet_convert_caffe import DarknetConvertCaffe
from easy_converter.helper.arguments_parse import TaskArgumentsParse


if __name__ == '__main__':
    print("process start...")
    options = TaskArgumentsParse.darknet2caffe_parse_arguments()
    convert = DarknetConvertCaffe(options.cfg_path, options.weight_path)
    convert.convert_caffe()
    print("process end!")