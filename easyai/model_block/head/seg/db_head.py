#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.upsample_layer import DeConvBNActivationBlock
from easyai.model_block.utility.base_block import *


class BinarizeHead(BaseBlock):

    def __init__(self, in_channels,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.BinarizeHead)
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=in_channels // 4,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1,
                                           bnName=bn_name,
                                           activationName=activation_name)
        self.conv2 = DeConvBNActivationBlock(in_channels=in_channels // 4,
                                             out_channels=in_channels // 4,
                                             kernel_size=2,
                                             stride=2,
                                             bias=True,
                                             bn_name=bn_name,
                                             activation_name=activation_name)
        self.conv3 = nn.ConvTranspose2d(in_channels=in_channels // 4,
                                        out_channels=1,
                                        kernel_size=2,
                                        stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        return x


class DBHead(BaseBlock):
    """
        Differentiable Binarization (DB) for text detection:
            see https://arxiv.org/abs/1911.08947
        args:
            params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(HeadType.DBHead)
        self.k = k
        self.binarize = BinarizeHead(in_channels,
                                     bn_name=bn_name,
                                     activation_name=activation_name)
        self.thresh = BinarizeHead(in_channels,
                                   bn_name=bn_name,
                                   activation_name=activation_name)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps
        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        return y
