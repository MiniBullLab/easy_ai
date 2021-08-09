#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.residual_block import ResidualBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.backbone_registry import REGISTERED_CLS_BACKBONE


__all__ = ['ResNet18', 'ResNet18V2',
           'ResNet34', 'TextResNet34',
           'ResNet50', 'WideResnet50V2',
           'ResNet101', 'ResNet152']


class ResNet(BaseBackbone):
    def __init__(self, data_channel=3, num_blocks=(2, 2, 2, 2), out_channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), use_short=(0, 0, 0, 0),
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU,
                 block_flag=0, head_type=1,
                 first_stride=2, width_group=64):
        super().__init__(data_channel)
        self.set_name(BackboneName.ResNet18)
        self.num_blocks = num_blocks
        self.out_channels = out_channels
        self.strides = strides
        self.dilations = dilations
        self.use_short = use_short
        self.bn_name = bn_name
        self.activation_name = activation_name
        self.block_flag = block_flag
        self.first_output = 64
        self.in_channels = self.first_output
        self.head_type = head_type
        self.first_stride = first_stride
        self.width_group = width_group

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        if self.head_type == 0:
            layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                           out_channels=self.first_output,
                                           kernel_size=3,
                                           stride=self.first_stride,
                                           padding=1,
                                           bnName=self.bn_name,
                                           activationName=self.activation_name)
            self.add_block_list(layer1.get_name(), layer1, self.first_output)
        elif self.head_type == 1:
            layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                           out_channels=self.first_output,
                                           kernel_size=7,
                                           stride=self.first_stride,
                                           padding=3,
                                           bnName=self.bn_name,
                                           activationName=self.activation_name)
            self.add_block_list(layer1.get_name(), layer1, self.first_output)

            layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)
        elif self.head_type == 2:
            layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                           out_channels=32,
                                           kernel_size=3,
                                           stride=self.first_stride,
                                           padding=1,
                                           bnName=self.bn_name,
                                           activationName=self.activation_name)
            self.add_block_list(layer1.get_name(), layer1, 32)

            layer11 = ConvBNActivationBlock(in_channels=32,
                                            out_channels=32,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bnName=self.bn_name,
                                            activationName=self.activation_name)
            self.add_block_list(layer11.get_name(), layer11, 32)

            layer12 = ConvBNActivationBlock(in_channels=32,
                                            out_channels=self.first_output,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bnName=self.bn_name,
                                            activationName=self.activation_name)
            self.add_block_list(layer12.get_name(), layer12, self.first_output)

            layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.add_block_list(LayerType.MyMaxPool2d, layer2, self.first_output)

        self.in_channels = self.first_output
        for index, num_block in enumerate(self.num_blocks):
            self.make_resnet_block(self.out_channels[index], self.num_blocks[index],
                                   self.strides[index], self.dilations[index],
                                   self.use_short[index], self.bn_name, self.activation_name,
                                   self.block_flag)
            self.in_channels = self.block_out_channels[-1]

    def make_resnet_block(self, out_channels, num_block, stride, dilation,
                          use_short, bn_name, activation, block_flag):
        expansion = 0
        if block_flag == 0:
            expansion = 1
        elif block_flag == 1:
            expansion = 4
        elif block_flag == 2:
            expansion = 4
        down_layers = ResidualBlock(self.block_flag, self.in_channels, out_channels, stride,
                                    dilation=dilation, expansion=expansion,
                                    use_short=use_short,
                                    base_width=self.width_group,
                                    bn_name=bn_name,
                                    activation_name=activation)
        name = "down_%s" % down_layers.get_name()
        temp_output_channel = out_channels * expansion
        self.add_block_list(name, down_layers, temp_output_channel)
        for i in range(num_block - 1):
            layer = ResidualBlock(self.block_flag, temp_output_channel, out_channels,
                                  expansion=expansion, base_width=self.width_group,
                                  bn_name=bn_name, activation_name=activation)
            temp_output_channel = out_channels * expansion
            self.add_block_list(layer.get_name(), layer, temp_output_channel)

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet18)
class ResNet18(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[2, 2, 2, 2],
                         block_flag=0)
        self.set_name(BackboneName.ResNet18)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet18V2)
class ResNet18V2(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[2, 2, 2, 2],
                         strides=(1, 2, 2, 2),
                         use_short=(1, 2, 2, 2),
                         block_flag=0,
                         head_type=2)
        self.set_name(BackboneName.ResNet18V2)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet34)
class ResNet34(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 4, 6, 3],
                         block_flag=0)
        self.set_name(BackboneName.ResNet34)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet50)
class ResNet50(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 4, 6, 3],
                         block_flag=1)
        self.set_name(BackboneName.ResNet50)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet101)
class ResNet101(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 4, 23, 3],
                         block_flag=1)
        self.set_name(BackboneName.ResNet101)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.ResNet152)
class ResNet152(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 8, 36, 3],
                         block_flag=1)
        self.set_name(BackboneName.ResNet152)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.WideResnet50V2)
class WideResnet50V2(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 4, 6, 3],
                         head_type=1,
                         block_flag=2,
                         width_group=64 * 2)
        self.set_name(BackboneName.WideResnet50V2)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.TextResNet34)
class TextResNet34(ResNet):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         num_blocks=[3, 4, 6, 3],
                         strides=(1, (2, 1), (2, 1), (2, 1)),
                         use_short=(1, 2, 2, 2),
                         block_flag=0,
                         head_type=2,
                         first_stride=1)
        self.set_name(BackboneName.TextResNet34)


