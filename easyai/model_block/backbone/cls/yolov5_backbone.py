#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.cls.yolov5_block import FocusBlock, C3Block, SpatialPyramidPoolingWithConv
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE


__all__ = ['Yolov5Net', 'Yolov5s_Backbone', 'Yolov5m_Backbone',
           'Yolov5l_Backbone', 'Yolov5x_Backbone']


class Yolov5Net(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=(1, 3, 3, 1),
                 out_channels=(64, 128, 256, 512), strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1), bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.Swish):
        super().__init__(data_channel)
        self.set_name(BackboneName.Yolov5Net)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.activationName = activationName
        self.bnName = bnName
        self.first_output = int(out_channels[0] / 2)
        self.in_channel = self.first_output

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = FocusBlock(in_channels=self.data_channel,
                            out_channels=self.first_output,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=1,
                            bnName=self.bnName,
                            activationName=self.activationName)
        self.add_block_list(layer1.get_name(), layer1, self.first_output)

        for index, num_block in enumerate(self.num_blocks):
            self.make_yolov5_layer(self.out_channels[index], self.num_blocks[index],
                                    self.strides[index], self.dilations[index],
                                    self.bnName, self.activationName, index)
            self.in_channel = self.block_out_channels[-1]

    def make_yolov5_layer(self, out_channel, num_block, stride, dilation,
                          bnName, activationName, index):

        down_layers = ConvBNActivationBlock(in_channels=self.in_channel,
                                            out_channels=out_channel,
                                            kernel_size=3,
                                            stride=stride,
                                            padding=dilation,
                                            dilation=dilation,
                                            bnName=bnName,
                                            activationName=activationName)
        name = "down_%s" % down_layers.get_name()
        self.add_block_list(name, down_layers, out_channel)

        if index == len(self.num_blocks) - 1:
            shortcut = False

            spp_layer = SpatialPyramidPoolingWithConv(in_channels=out_channel, out_channels=out_channel,
                                                      pool_sizes=(5, 9, 13))
            self.add_block_list(spp_layer.get_name(), spp_layer, out_channel)
        else:
            shortcut = True

        c3_block = C3Block(in_channels=out_channel,
                           out_channels=out_channel,
                           number=num_block,
                           shortcut=shortcut,
                           groups=1,
                           bnName=bnName,
                           activationName=activationName,
                           expansion = 0.5)
        self.add_block_list(c3_block.get_name(), c3_block, out_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Yolov5s_Backbone)
class Yolov5s_Backbone(Yolov5Net):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[1, 3, 3, 1],
                         out_channels=(64, 128, 256, 512))
        self.set_name(BackboneName.Yolov5s_Backbone)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Yolov5m_Backbone)
class Yolov5m_Backbone(Yolov5Net):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[2, 6, 6, 2],
                         dilations=[96, 192, 384, 768])
        self.set_name(BackboneName.Yolov5m_Backbone)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Yolov5l_Backbone)
class Yolov5l_Backbone(Yolov5Net):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 9, 9, 3],
                         dilations=[128, 256, 512, 1024])
        self.set_name(BackboneName.Yolov5l_Backbone)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Yolov5x_Backbone)
class Yolov5x_Backbone(Yolov5Net):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[4, 12, 12, 4],
                         dilations=[160, 320, 640, 1280])
        self.set_name(BackboneName.Yolov5x_Backbone)


