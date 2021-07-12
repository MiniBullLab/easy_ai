# import os
# import sys
# sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.model_block.utility.backbone_factory import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_onnx.model_show import ModelShow
from easyai.helper.arguments_parse import ToolArgumentsParse


class ModelNetShow():

    def __init__(self):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()
        self.show_process = ModelShow()

    def model_show(self, model_name):
        input_x = torch.randn(1, 3, 32, 128)
        self.show_process.set_input(input_x)
        model_config = {"type": model_name}
        model = self.model_factory.get_model(model_config)
        self.show_process.show_from_model(model)

    def backbone_show(self, backbone_path):
        input_x = torch.randn(1, 3, 128, 128)
        self.show_process.set_input(input_x)
        model_config = {'type': backbone_path}
        model = self.backbone_factory.get_backbone_model(model_config)
        self.show_process.show_from_model(model)

    def onnx_show(self, onnx_path):
        self.show_process.show_from_onnx(onnx_path)


def main(options_param):
    show = ModelNetShow()
    if options_param.model is not None:
        show.model_show(options.model)
    elif options_param.backbone is not None:
        show.backbone_show(options.backbone)
    elif options_param.onnx_path is not None:
        show.onnx_show(options.onnx_path)


if __name__ == '__main__':
    options = ToolArgumentsParse.model_show_parse()
    main(options)
