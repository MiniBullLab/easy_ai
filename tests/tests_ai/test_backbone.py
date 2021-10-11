#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

import os
import sys
sys.path.insert(0, os.getcwd() + "/..")
import torch
from easyai.name_manager import BackboneName
from easyai.model_block.backbone.utility import BackboneFactory


def main():
    backbone_factory = BackboneFactory()
    input_x = torch.randn(1, 3, 32, 32)
    model = backbone_factory.get_base_model(BackboneName.Attention92)
    output = model(input_x)
    print(output)


if __name__ == '__main__':
    main()
