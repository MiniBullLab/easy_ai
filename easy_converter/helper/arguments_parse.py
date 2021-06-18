#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from optparse import OptionParser
import argparse


class TaskArgumentsParse():

    def __init__(self):
        pass

    @classmethod
    def darknet2caffe_parse_arguments(cls):
        parser = OptionParser()
        parser.description = "This program is keras convert to onnx"

        parser.add_option("-c", "--cfg", dest="cfg_path",
                          metavar="PATH", type="string", default=None,
                          help="darknet cfg path")

        parser.add_option("-w", "--weight", dest="weight_path",
                          type="string", default=None,
                          help="darknet weight path")

        (options, args) = parser.parse_args()

        return options

    @classmethod
    def keras2onnx_parse_arguments(cls):
        parser = OptionParser()
        parser.description = "This program is keras convert to onnx"

        parser.add_option("-i", "--h5", dest="h5_path",
                          metavar="PATH", type="string", default=None,
                          help="keras h5 model path")

        parser.add_option("-m", "--model", dest="model_name",
                          type="string", default=None,
                          help="keras model name")

        (options, args) = parser.parse_args()

        return options

    @classmethod
    def onnx2caffe_parse_arguments(cls):
        parser = OptionParser()
        parser.description = "This program is onnx convert to caffe"

        parser.add_option("-i", "--input", dest="input_path",
                          metavar="PATH", type="string", default=None,
                          help="onnx path")

        parser.add_option("-p", "--proto", dest="proto_path",
                          metavar="PATH", type="string", default=None,
                          help="prototxt path")

        (options, args) = parser.parse_args()

        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")
            else:
                options.input_path = os.path.normpath(options.input_path)
        else:
            parser.error("'input' option is required to run this program")

        return options

    @classmethod
    def onnx2simple_parse_arguments(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('input_model', help='Input ONNX model')
        parser.add_argument('output_model', help='Output ONNX model')
        parser.add_argument('check_n', help='Check whether the output is correct with n random inputs',
                            nargs='?', type=int, default=3)
        parser.add_argument('--enable-fuse-bn',
                            help='Enable ONNX fuse_bn_into_conv optimizer. '
                                 'In some cases it causes incorrect model (https://github.com/onnx/onnx/issues/2677).',
                            action='store_true')
        parser.add_argument('--skip-fuse-bn', help='This argument is deprecated. Fuse-bn has been skippped by default',
                            action='store_true')
        parser.add_argument('--skip-optimization', help='Skip optimization of ONNX optimizers.',
                            action='store_true')
        parser.add_argument(
            '--input-shape',
            help='The manually-set static input shape, '
                 'useful when the input shape is dynamic. '
                 'The value should be "input_name:dim0,dim1,...,dimN" or '
                 'simply "dim0,dim1,...,dimN" when there is only one input, '
                 'for example, "data:1,3,224,224" or "1,3,224,224". '
                 'Note: you might want to use some visualization tools '
                 'like netron to make sure what the input name and dimension ordering (NCHW or NHWC) is.',
            type=str, nargs='+')
        args = parser.parse_args()
        return args

    @classmethod
    def onnx2tensorflow_parse_arguments(cls):
        parser = OptionParser()
        parser.description = "This program is onnx convert to tensorflow"

        parser.add_option("-i", "--input", dest="input_path",
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

        return options

    @classmethod
    def tensorflow2onnx_parse_arguments(cls):
        parser = OptionParser()
        parser.description = "This program is tensorflow convert to onnx"

        parser.add_option("-w", "--pb", dest="pb_path",
                          metavar="PATH", type="string", default=None,
                          help="tensorflow pb path")

        (options, args) = parser.parse_args()

        if options.pb_path:
            if not os.path.exists(options.pb_path):
                parser.error("Could not find the onnx file")
            else:
                options.pb_path = os.path.normpath(options.pb_path)
        else:
            parser.error("'pb_path' option is required to run this program")

        return options
