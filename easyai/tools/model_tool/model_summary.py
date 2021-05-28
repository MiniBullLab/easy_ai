#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
from easyai.model_block.backbone.utility import BackboneFactory
from easyai.model.utility.model_factory import ModelFactory
from easyai.torch_utility.torch_summary import summary
from easyai.helper.arguments_parse import ToolArgumentsParse


class ModelSummary():

    def __init__(self):
        self.backbone_factory = BackboneFactory()
        self.model_factory = ModelFactory()

    def backbone_model(self, backbone_name):
        input_x = (1, 3, 224, 224)
        model_config = {"type": backbone_name,
                        "data_channel": 3}
        backbone = self.backbone_factory.get_backbone_model(model_config)
        summary(backbone, input_x)

    def model(self, model_name):
        input_x = (1, 3, 224, 224)
        model_config = {"type": model_name,
                        "data_channel": 3}
        model = self.model_factory.get_model(model_config)
        summary(model, input_x)


def main(options_param):
    print("process start...")
    test = ModelSummary()
    if options_param.model is not None:
        test.model(options.model)
    elif options_param.backbone is not None:
        test.backbone_model(options.backbone)
    print("process end!")


if __name__ == '__main__':
    options = ToolArgumentsParse.model_parse()
    main(options)
