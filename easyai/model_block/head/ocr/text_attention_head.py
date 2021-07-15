#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.upsample_layer import DeConvBNActivationBlock
from easyai.model_block.utility.base_block import *


class TextAttentionHead(BaseBlock):

    def __init__(self):
        super().__init__(HeadType.TextAttentionHead)