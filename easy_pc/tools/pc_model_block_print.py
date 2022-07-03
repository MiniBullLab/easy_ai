#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger
from easy_pc.model.utility.pc_model_factory import PCModelFactory
if EasyLogger.check_init():
    log_file_path = EasyLogger.get_log_file_path("pc_tools.log")
    EasyLogger.init(logfile_level="debug", log_file=log_file_path, stdout_level="error")

import torch
from easy_pc.model_block.utility.pc_backbone_factory import PCBackboneFactory

from easyai.helper.arguments_parse import ToolArgumentsParse


class PCModelBlockPrint():

    def __init__(self):
        self.backbone_factory = PCBackboneFactory()
        self.model_factory = PCModelFactory()

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
    show = PCModelBlockPrint()
    if options_param.model is not None:
        show.model_print(options.model)
    elif options_param.backbone is not None:
        show.backbone_model_print(options.backbone)


if __name__ == '__main__':
    options = ToolArgumentsParse.model_parse()
    main(options)
