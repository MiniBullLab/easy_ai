#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import argparse
import sys
import onnx
import onnxsim


def parse_arguments():
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


def main(args):
    print("Simplifying...")
    input_shapes = {}
    if args.input_shape is not None:
        for x in args.input_shape:
            if ':' not in x:
                input_shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                input_shapes[name] = shape
    model_opt, check_ok = onnxsim.simplify(
        args.input_model, check_n=args.check_n,
        perform_optimization=not args.skip_optimization,
        skip_fuse_bn=not args.enable_fuse_bn, input_shapes=input_shapes)

    onnx.save(model_opt, args.output_model)

    if check_ok:
        print("Ok!")
    else:
        print("Check failed, please be careful to use the simplified model, "
              "or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" "
              "(run \"python3 -m onnxsim -h\" for details)")
        sys.exit(1)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)

