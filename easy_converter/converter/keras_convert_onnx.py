#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import pathlib
import onnx
from keras2onnx import convert_keras
from keras import models


class KerasConvertOnnx():

    def __init__(self, h5_model_path):
        self.target_opset = 10
        self.h5_model_path = pathlib.Path(h5_model_path)
        self.onnx_save_path = self.h5_model_path.with_suffix(".onnx")

    def convert_onnx_from_h5(self, net_name):
        # get model struct and weights
        keras_model = models.load_model(str(self.h5_model_path))
        # onnx_model = onnxmltools.convert_keras(keras_model)
        onnx_model = convert_keras(keras_model, net_name,
                                   target_opset=self.target_opset,
                                   channel_first_inputs=['net_input'])
        onnx.save_model(onnx_model, str(self.onnx_save_path))


