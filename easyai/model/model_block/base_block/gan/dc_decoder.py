#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import ActivationType, NormalizationType
from easyai.base_name.block_name import BlockType, LayerType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.model.model_block.base_block.utility.upsample_layer import DeConvBNActivationBlock
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock


class DCDecoder(BaseBlock):

    def __init__(self, csize, in_channel=100, out_channel=64,
                 final_out_channel=3, n_extra_layers=0,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(BlockType.DCDecoder)
        assert csize % 16 == 0, "csize has to be a multiple of 16"
        self.block = nn.Sequential()

        out_channel, tisize = out_channel // 2, 4
        while tisize != csize:
            out_channel = out_channel * 2
            tisize = tisize * 2

        deconv = DeConvBNActivationBlock(in_channels=in_channel,
                                         out_channels=out_channel,
                                         kernel_size=4,
                                         stride=1,
                                         padding=0,
                                         bias=False,
                                         bn_name=bn_name,
                                         activation_name=activation_name)
        self.block.add_module(deconv.get_name(), deconv)

        in_size = 4
        while in_size < csize // 2:
            in_feat = out_channel
            out_feat = out_channel // 2
            pyramid_deconv = DeConvBNActivationBlock(in_channels=in_feat,
                                                     out_channels=out_feat,
                                                     kernel_size=4,
                                                     stride=2,
                                                     padding=1,
                                                     bias=False,
                                                     bn_name=bn_name,
                                                     activation_name=activation_name)
            temp_name = "pyramid_%s_%d" % (pyramid_deconv.get_name(), csize)
            self.block.add_module(temp_name, pyramid_deconv)
            out_channel = out_channel // 2
            in_size = in_size * 2

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

        final_deconv = DeConvBNActivationBlock(in_channels=out_channel,
                                               out_channels=final_out_channel,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1,
                                               bias=False,
                                               bn_name=NormalizationType.EmptyNormalization,
                                               activation_name=ActivationType.Tanh)
        temp_name = "final_%s" % final_deconv.get_name()
        self.block.add_module(temp_name, final_deconv)

    def forward(self, x):
        x = self.block(x)
        return x

    def forward(self, x):
        x = self.block(x)
        return x
