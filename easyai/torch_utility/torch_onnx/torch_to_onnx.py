#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import pathlib
import os.path
import torch
from torch import onnx
from easyai.torch_utility.torch_model_process import TorchModelProcess


# # Override Upsample's ONNX export until new opset is supported
# @torch.onnx.symbolic.parse_args('v', 'is')
# def upsample_nearest2d(g, input, output_size):
#     height_scale = float(output_size[-2]) / input.type().sizes()[-2]
#     width_scale = float(output_size[-1]) / input.type().sizes()[-1]
#     return g.op("Upsample", input,
#                 scales_f=(1, 1, height_scale, width_scale),
#                 mode_s="nearest")
#
#
# torch.onnx.symbolic.upsample_nearest2d = upsample_nearest2d
#
#
# @torch.onnx.symbolic.parse_args('v', 'is', 'i')
# def upsample_bilinear2d(g, input, output_size):
#     height_scale = float(output_size[-2]) / input.type().sizes()[-2]  # 8
#     width_scale = float(output_size[-1]) / input.type().sizes()[-1]  # 8
#     return g.op("Upsample", input,
#                 scales_f=(1, 1, height_scale, width_scale),
#                 mode_s="linear")
#
#
# torch.onnx.symbolic.upsample_bilinear2d = upsample_bilinear2d


class TorchConvertOnnx():

    def __init__(self, channel=3, width=224, height=224):
        self.model_process = TorchModelProcess()
        self.input_x = torch.ones(1, channel, width, height)
        self.save_dir = "."
        self.opset = 9
        self.input_names = None
        self.output_names = None

    def set_input(self, input_torch):
        self.input_x = input_torch

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def set_input_names(self, names):
        self.input_names = names

    def set_output_names(self, names):
        self.output_names = names

    def torch2onnx(self, model, weight_path=None):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if weight_path is not None:
            temp_path = pathlib.Path(weight_path)
            temp_path = temp_path.with_suffix(".native")
            if temp_path.exists():
                model = torch.jit.load(str(temp_path))
            else:
                self.model_process.load_latest_model(weight_path, model)
        save_onnx_path = os.path.join(self.save_dir, "%s.onnx" % model.get_name())
        onnx.export(model, self.input_x, save_onnx_path,
                    export_params=True, verbose=False,
                    input_names=self.input_names, output_names=self.output_names)
        return save_onnx_path
