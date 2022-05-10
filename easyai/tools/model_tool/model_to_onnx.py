#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torch
from easyai.model_block.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.torch_to_onnx import TorchConvertOnnx
from easyai.config.utility.config_factory import ConfigFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class ModelConverter():

    def __init__(self, input_size=(352, 640)):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.converter = TorchConvertOnnx()
        self.input_size = input_size  # w * h

    def convert_process(self, net_config, weight_path, save_dir,
                        input_names=None, output_names=None):
        self.converter.set_input_names(input_names)
        self.converter.set_output_names(output_names)
        onnx_path = self.model_convert(net_config, weight_path, save_dir)
        return onnx_path

    def model_convert(self, net_config, weight_path, save_dir):
        self.converter.set_save_dir(save_dir)
        model = self.model_factory.get_model(net_config)
        data_channel = model.get_data_channel()
        input_x = torch.randn(1, data_channel, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        save_onnx_path = self.converter.torch2onnx(model, weight_path)
        return save_onnx_path

    def base_model_convert(self, net_config, weight_path, save_dir):
        self.converter.set_save_dir(save_dir)
        model = self.backbone_factory.get_backbone_model(net_config)
        data_channel = model.get_data_channel()
        input_x = torch.randn(1, data_channel, self.input_size[1], self.input_size[0])
        self.converter.set_input(input_x)
        save_onnx_path = self.converter.torch2onnx(model, weight_path)
        return save_onnx_path


def main(options_param):
    config_factory = ConfigFactory()
    task_config = config_factory.get_config(options_param.task_name, config_path=options_param.config_path)

    converter = ModelConverter(task_config.data["image_size"])
    if options_param.model is not None:
        if options_param.config_path is not None:
            model_config = {"type": options_param.model,
                            "data_channel": 3,
                            "class_number": len(task_config.class_name)}
        else:
            model_config = {"type": options_param.model,
                            "data_channel": 3}
        converter.model_convert(model_config, options_param.weight_path, options_param.save_dir)
    elif options_param.backbone is not None:
        model_config = {"type": options_param.backbone,
                        "data_channel": 3}
        converter.base_model_convert(model_config, options_param.weight_path, options_param.save_dir)


if __name__ == '__main__':
    options = ToolArgumentsParse.model_convert_parse()
    main(options)

