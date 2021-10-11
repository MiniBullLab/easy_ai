#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import pathlib
import onnx
from onnx_tf import backend


class OnnxConvertTensorflow():

    def __init__(self, onnx_path):
        self.target_opset = 9
        self.onnx_path = pathlib.Path(onnx_path)
        self.tensorflow_model_save_path = self.onnx_path.with_suffix(".pb")

    def convert_tensorflow(self):
        model = onnx.load(str(self.onnx_path))
        tf_rep = backend.prepare(model, optset_version=self.target_opset)
        self.print_param(tf_rep)
        tf_rep.export_graph(str(self.tensorflow_model_save_path))

    def print_param(self, tf_rep):
        # Input nodes to the model
        print('inputs:', tf_rep.inputs)

        # Output nodes from the model
        print('outputs:', tf_rep.outputs)

        # All nodes in the model
        print('tensor_dict:')
        print(tf_rep.tensor_dict)

