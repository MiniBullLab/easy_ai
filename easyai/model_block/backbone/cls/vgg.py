#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.model_block.base_block.common.pooling_layer import MyMaxPool2d
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.utility_block import ConvDropBNActivationBlock
from easyai.model_block.base_block.common.utility_block import ConvActivationBlock
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.backbone_registry import REGISTERED_CLS_BACKBONE


__all__ = ['VGG11', 'VGG13', 'VGG16', 'VGG19', 'TextVGG']


class VGG(BaseBackbone):

    CONFIG_DICT = {
        BackboneName.Vgg11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        BackboneName.Vgg13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
                             512, 512, 'M', 512, 512, 'M'],
        BackboneName.Vgg16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                             512, 512, 512, 'M', 512, 512, 512, 'M'],
        BackboneName.Vgg19: [64, 64, 'M', 128, 128, 'M',
                             256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                             512, 512, 512, 512, 'M'],
        'vgg13_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512],
        'vgg16_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512],
        'vgg19_dilated8': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512],
        'vgg13_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
        'vgg16_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512],
        'vgg19_dilated16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512],
    }

    def __init__(self, data_channel=3, vgg_name=BackboneName.Vgg19,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.ReLU, is_norm=False):
        super().__init__(data_channel)
        self.set_name(BackboneName.Vgg19)
        self.activationName = activationName
        self.bnName = bnName
        self.is_norm = is_norm
        self.vgg_cfg = VGG.CONFIG_DICT.get(vgg_name, None)
        self.create_block_list()

    def create_block_list(self):
        self.block_out_channels = []
        self.index = 0

        self.make_layers(self.vgg_cfg)

    def make_layers(self, cfg):
        in_channels = self.data_channel
        for v in cfg:
            if v == 'M':
                max_pooling = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                self.add_block_list(LayerType.MyMaxPool2d, max_pooling, in_channels)
            else:
                if self.is_norm:
                    conv2d = ConvBNActivationBlock(in_channels=in_channels,
                                                   out_channels=v,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   bnName=self.bnName,
                                                   activationName=self.activationName)
                    self.add_block_list(conv2d.get_name(), conv2d, v)
                else:
                    conv2d = ConvActivationBlock(in_channels=in_channels,
                                                 out_channels=v,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 dilation=1,
                                                 activationName=self.activationName)
                    self.add_block_list(conv2d.get_name(), conv2d, v)
                in_channels = v

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.TextVGG)
class TextVGG(BaseBackbone):

    def __init__(self, data_channel=3,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.ReLU):
        super().__init__(data_channel)
        self.set_name(BackboneName.Vgg19)
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.create_block_list()

    def create_block_list(self):
        in_channels = self.data_channel
        conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                      out_channels=50,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 50)

        conv2 = nn.Conv2d(self.block_out_channels[-1], 100,
                          stride=1, kernel_size=3, padding=1)
        self.add_block_list(LayerType.Convolutional, conv2, 100)

        drop1 = nn.Dropout(p=0.1)
        self.add_block_list(LayerType.Dropout, drop1, self.block_out_channels[-1])

        conv3 = ConvDropBNActivationBlock(p=0.1,
                                          in_channels=self.block_out_channels[-1],
                                          out_channels=100,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 100)

        pooling1 = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.add_block_list(pooling1.get_name(), pooling1, self.block_out_channels[-1])

        conv4 = nn.Conv2d(self.block_out_channels[-1], 200,
                          stride=1, kernel_size=3, padding=1)
        self.add_block_list(LayerType.Convolutional, conv4, 200)

        drop2 = nn.Dropout(p=0.2)
        self.add_block_list(LayerType.Dropout, drop2, self.block_out_channels[-1])

        conv5 = ConvDropBNActivationBlock(p=0.2,
                                          in_channels=self.block_out_channels[-1],
                                          out_channels=200,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(conv5.get_name(), conv5, 200)

        pooling2 = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.add_block_list(pooling2.get_name(), pooling2, self.block_out_channels[-1])

        conv6 = ConvDropBNActivationBlock(p=0.3,
                                          in_channels=self.block_out_channels[-1],
                                          out_channels=250,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(conv6.get_name(), conv6, 250)

        conv7 = nn.Conv2d(self.block_out_channels[-1], 300,
                          stride=1, kernel_size=3, padding=1)
        self.add_block_list(LayerType.Convolutional, conv7, 300)

        drop3 = nn.Dropout(p=0.3)
        self.add_block_list(LayerType.Dropout, drop3, self.block_out_channels[-1])

        conv8 = ConvDropBNActivationBlock(p=0.3,
                                          in_channels=self.block_out_channels[-1],
                                          out_channels=300,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(conv8.get_name(), conv8, 300)

        pooling3 = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.add_block_list(pooling3.get_name(), pooling3, self.block_out_channels[-1])

        conv9 = ConvDropBNActivationBlock(p=0.4,
                                          in_channels=self.block_out_channels[-1],
                                          out_channels=350,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(conv9.get_name(), conv9, 350)

        conv10 = nn.Conv2d(self.block_out_channels[-1], 400,
                           stride=1, kernel_size=3, padding=1)
        self.add_block_list(LayerType.Convolutional, conv10, 400)

        drop4 = nn.Dropout(p=0.4)
        self.add_block_list(LayerType.Dropout, drop4, self.block_out_channels[-1])

        conv11 = ConvDropBNActivationBlock(p=0.4,
                                           in_channels=self.block_out_channels[-1],
                                           out_channels=400,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bn_name=self.bn_name,
                                           activation_name=self.activation_name)
        self.add_block_list(conv11.get_name(), conv11, 400)

        pooling4 = MyMaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.add_block_list(pooling4.get_name(), pooling4, self.block_out_channels[-1])

    def forward(self, x):
        output_list = []
        for block in self._modules.values():
            x = block(x)
            output_list.append(x)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg11)
class VGG11(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg11)
        self.set_name(BackboneName.Vgg11)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg13)
class VGG13(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg13)
        self.set_name(BackboneName.Vgg13)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg16)
class VGG16(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg16)
        self.set_name(BackboneName.Vgg16)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.Vgg19)
class VGG19(VGG):

    def __init__(self, data_channel):
        super().__init__(data_channel=data_channel,
                         vgg_name=BackboneName.Vgg19)
        self.set_name(BackboneName.Vgg19)
