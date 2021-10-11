#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import ActivationType, NormalizationType
from easyai.name_manager.block_name import BlockType, LayerType
from easyai.model_block.utility.base_block import *
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.utility_block import ConvActivationBlock


class DCEncoder(BaseBlock):

    def __init__(self, csize, in_channel=3, out_channel=64,
                 final_out_channel=100, n_extra_layers=0,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.LeakyReLU):
        super().__init__(BlockType.DCEncoder)
        assert csize % 16 == 0,  "csize(%d) has to be a multiple of 16" % csize
        csize = csize / 2
        self.block = nn.Sequential()

        conv = ConvActivationBlock(in_channels=in_channel,
                                   out_channels=out_channel,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False,
                                   activationName=activation_name)
        self.block.add_module(conv.get_name(), conv)

        # Extra layers
        for t in range(n_extra_layers):
            extra_conv = ConvBNActivationBlock(in_channels=out_channel,
                                               out_channels=out_channel,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               bias=False,
                                               bnName=bn_name,
                                               activationName=activation_name)
            temp_name = "extra_%s_%d" % (extra_conv.get_name(), t)
            self.block.add_module(temp_name, extra_conv)

        while csize > 4:
            in_feat = out_channel
            out_feat = out_channel * 2
            pyramid_conv = ConvBNActivationBlock(in_channels=in_feat,
                                                 out_channels=out_feat,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False,
                                                 bnName=bn_name,
                                                 activationName=activation_name)
            temp_name = "pyramid_%s_%d" % (pyramid_conv.get_name(), csize)
            self.block.add_module(temp_name, pyramid_conv)
            out_channel = out_channel * 2
            csize = csize / 2

        self.output_channel = out_channel
        self.block.add_module('final_{0}'.format(LayerType.Convolutional),
                              nn.Conv2d(self.output_channel, final_out_channel, 4, 1, 0, bias=False))

    def get_output_channel(self):
        return self.output_channel

    def forward(self, x):
        for layer in self.block.children():
            x = layer(x)
            # print(x.shape)
        return x
