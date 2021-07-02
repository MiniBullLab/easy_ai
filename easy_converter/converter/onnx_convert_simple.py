#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import sys
import onnx
import onnxsim


class OnnxConvertSimple():

    def __init__(self, onnx_path, check_n=3,
                 skip_optimization=True, enable_fuse_bn=True):
        self.onnx_path = onnx_path
        self.input_shape = None
        self.check_n = check_n
        self.skip_optimization = skip_optimization
        self.enable_fuse_bn = enable_fuse_bn

    def convert_simple(self, output_path=None):
        input_shapes = {}
        if self.input_shape is not None:
            for x in self.input_shape:
                if ':' not in x:
                    input_shapes[None] = list(map(int, x.split(',')))
                else:
                    pieces = x.split(':')
                    # for the input name like input:0
                    name, shape = ':'.join(
                        pieces[:-1]), list(map(int, pieces[-1].split(',')))
                    input_shapes[name] = shape
        model_opt, check_ok = onnxsim.simplify(
            self.onnx_path, check_n=self.check_n,
            perform_optimization=not self.skip_optimization,
            skip_fuse_bn=not self.enable_fuse_bn, input_shapes=input_shapes)

        onnx.save(model_opt, output_path)

        if check_ok:
            print("Ok!")
        else:
            print("Check failed, please be careful to use the simplified model, "
                  "or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" "
                  "(run \"python3 -m onnxsim -h\" for details)")