#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.model_block.backbone.utility import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.helper.arguments_parse import ToolArgumentsParse


class ModelBlockPrint():

    def __init__(self):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()

    def backbone_model_print(self, backbone_name):
        input_x = torch.randn(1, 3, 224, 224)
        model_config = {"type": backbone_name}
        backbone = self.backbone_factory.get_backbone_model(model_config)
        if backbone is not None:
            backbone.print_block_name()
            # for k, value in backbone.named_parameters():
            #     print(k, value)

    def model_print(self, model_name):
        input_x = torch.randn(1, 3, 224, 224)
        model_config = {"type": model_name}
        model = self.model_factory.get_model(model_config)
        if model is not None:
            model.print_block_name()
            # for k, value in model.named_children():
            #     print(k, value)


def main(options_param):
    show = ModelBlockPrint()
    if options_param.model is not None:
        show.model_print(options.backbone)
    elif options_param.backbone is not None:
        show.backbone_model_print(options.backbone)


if __name__ == '__main__':
    options = ToolArgumentsParse.model_parse()
    main(options)
