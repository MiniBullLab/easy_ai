#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easy_converter.converter.onnx_convert_simple import OnnxConvertSimple
from easy_converter.helper.arguments_parse import TaskArgumentsParse


if __name__ == '__main__':
    args = TaskArgumentsParse.onnx2simple_parse_arguments()
    test = OnnxConvertSimple(args.input_model,
                             args.check_n,
                             args.skip_optimization,
                             args.enable_fuse_bn)
    test.convert_simple(args.output_model)