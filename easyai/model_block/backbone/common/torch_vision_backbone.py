#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import torchvision
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.utility.backbone_registry import REGISTERED_VISION_BACKBONE

REGISTERED_VISION_BACKBONE.add_module(torchvision.models.resnet50(), BackboneName.VisionResnet50)
