#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.backbone_name import BackboneName
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.common.residual_block import InvertedResidualV2
from easyai.model_block.utility.base_backbone import *
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE

__all__ = ['MobileNetV3Large', 'MobileNetV3Small',
           "MobileNetV3LargeV05", "MobileNetV3SmallV05",
           "MobileNetV3SmallDown16"]


class MobileNetV3(BaseBackbone):

    def __init__(self, cfgs, mode, data_channel=3, scale=1.0,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.HardSwish):
        super().__init__(data_channel)
        self.set_name(BackboneName.MobileNetv3_small)
        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert mode in ['large', 'small']
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        self.cfgs = cfgs
        self.mode = mode
        self.scale = scale
        self.activation_name = activation_name
        self.bn_name = bn_name
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        # building first layer
        output_channel = self.make_divisible(16 * self.scale, 8)

        layer1 = ConvBNActivationBlock(in_channels=self.data_channel,
                                       out_channels=output_channel,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        input_channel = output_channel
        output_channel = self.make_layer(input_channel, self.cfgs)

        # building last several layers
        input_channel = self.block_out_channels[-1]
        layer2 = ConvBNActivationBlock(in_channels=input_channel,
                                       out_channels=output_channel,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, output_channel)

    def make_layer(self, input_channel, cfgs):
        # building inverted residual blocks
        hidden_channel = 0
        for flag, k, exp_size, c, use_se, use_hs, s in cfgs:
            # print(flag, k, exp_size, c, use_se, use_hs, s)
            output_channel = self.make_divisible(c * self.scale, 8)
            hidden_channel = self.make_divisible(exp_size * self.scale, 8)
            # print(input_channel, hidden_channel, hidden_channel)
            if use_hs == 0:
                temp_block = InvertedResidualV2(flag, input_channel, hidden_channel, output_channel,
                                                k, s, use_se,
                                                bn_name=self.bn_name,
                                                activation_name=ActivationType.ReLU)
            else:
                temp_block = InvertedResidualV2(flag, input_channel, hidden_channel, output_channel,
                                                k, s, use_se,
                                                bn_name=self.bn_name,
                                                activation_name=self.activation_name)
            self.add_block_list(temp_block.get_name(), temp_block, output_channel)
            input_channel = output_channel
        return hidden_channel

    def make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        output_list = []
        for key, block in self._modules.items():
            x = block(x)
            output_list.append(x)
            # print(key, x.shape)
        return output_list


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetv3_large)
class MobileNetV3Large(MobileNetV3):
    """
            Constructs a MobileNetV3-Large model
        """
    cfgs = [
        # f, k, t, c, SE, NL, s
        [1, 3, 16, 16, 0, 0, 1],
        [1, 3, 64, 24, 0, 0, 2],
        [1, 3, 72, 24, 0, 0, 1],
        [1, 5, 72, 40, 1, 0, 2],
        [1, 5, 120, 40, 1, 0, 1],
        [1, 5, 120, 40, 1, 0, 1],
        [1, 3, 240, 80, 0, 1, 2],
        [1, 3, 200, 80, 0, 1, 1],
        [1, 3, 184, 80, 0, 1, 1],
        [1, 3, 184, 80, 0, 1, 1],
        [1, 3, 480, 112, 1, 1, 1],
        [1, 3, 672, 112, 1, 1, 1],
        [1, 5, 672, 160, 1, 1, 1],
        [1, 5, 672, 160, 1, 1, 2],
        [1, 5, 960, 160, 1, 1, 1]
    ]

    def __init__(self, data_channel):
        super().__init__(MobileNetV3Large.cfgs, mode='large',
                         data_channel=data_channel)
        self.set_name(BackboneName.MobileNetv3_large)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV3_large_0_5)
class MobileNetV3LargeV05(MobileNetV3):
    """
            Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # f, k, t, c, SE, NL, s
        [1, 3, 16, 16, 0, 0, 1],
        [1, 3, 64, 24, 0, 0, 2],
        [1, 3, 72, 24, 0, 0, 1],
        [1, 5, 72, 40, 1, 0, 2],
        [1, 5, 120, 40, 1, 0, 1],
        [1, 5, 120, 40, 1, 0, 1],
        [1, 3, 240, 80, 0, 1, 2],
        [1, 3, 200, 80, 0, 1, 1],
        [1, 3, 184, 80, 0, 1, 1],
        [1, 3, 184, 80, 0, 1, 1],
        [1, 3, 480, 112, 1, 1, 1],
        [1, 3, 672, 112, 1, 1, 1],
        [1, 5, 672, 160, 1, 1, 1],
        [1, 5, 672, 160, 1, 1, 2],
        [1, 5, 960, 160, 1, 1, 1]
    ]

    def __init__(self, data_channel):
        super().__init__(MobileNetV3LargeV05.cfgs, mode='large',
                         scale=0.5, data_channel=data_channel)
        self.set_name(BackboneName.MobileNetV3_large_0_5)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetv3_small)
class MobileNetV3Small(MobileNetV3):
    """
        Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # f, k, t, c, SE, NL, s
        [1, 3, 16, 16, 1, 0, 2],
        [1, 3, 72, 24, 0, 0, 2],
        [1, 3, 88, 24, 0, 0, 1],
        [1, 5, 96, 40, 1, 1, 2],
        [1, 5, 240, 40, 1, 1, 1],
        [1, 5, 240, 40, 1, 1, 1],
        [1, 5, 120, 48, 1, 1, 1],
        [1, 5, 144, 48, 1, 1, 1],
        [1, 5, 288, 96, 1, 1, 2],
        [1, 5, 576, 96, 1, 1, 1],
        [1, 5, 576, 96, 1, 1, 1],
    ]

    def __init__(self, data_channel):
        super().__init__(MobileNetV3Small.cfgs, mode='small',
                         data_channel=data_channel)
        self.set_name(BackboneName.MobileNetv3_small)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV3_small_0_5)
class MobileNetV3SmallV05(MobileNetV3):
    """
        Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # f, k, t, c, SE, NL, s
        [1, 3, 16, 16, 1, 0, 2],
        [1, 3, 72, 24, 0, 0, 2],
        [1, 3, 88, 24, 0, 0, 1],
        [1, 5, 96, 40, 1, 1, 2],
        [1, 5, 240, 40, 1, 1, 1],
        [1, 5, 240, 40, 1, 1, 1],
        [1, 5, 120, 48, 1, 1, 1],
        [1, 5, 144, 48, 1, 1, 1],
        [1, 5, 288, 96, 1, 1, 2],
        [1, 5, 576, 96, 1, 1, 1],
        [1, 5, 576, 96, 1, 1, 1],
    ]

    def __init__(self, data_channel):
        super().__init__(MobileNetV3SmallV05.cfgs, mode='small',
                         scale=0.5, data_channel=data_channel)
        self.set_name(BackboneName.MobileNetV3_small_0_5)


@REGISTERED_CLS_BACKBONE.register_module(BackboneName.MobileNetV3SmallDown16)
class MobileNetV3SmallDown16(MobileNetV3):
    """
        Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # f, k, t, c, SE, NL, s
        [1, 3, 16, 16, 2, 0, (1, 1)],
        [1, 3, 72, 24, 0, 0, (2, 1)],
        [1, 3, 88, 24, 0, 0, 1],
        [1, 5, 96, 40, 2, 1, (2, 1)],
        [1, 5, 240, 40, 2, 1, 1],
        [1, 5, 240, 40, 2, 1, 1],
        [1, 5, 120, 48, 2, 1, 1],
        [1, 5, 144, 48, 2, 1, 1],
        [1, 5, 288, 96, 2, 1, (2, 1)],
        [1, 5, 576, 96, 2, 1, 1],
        [1, 5, 576, 96, 2, 1, 1],
    ]

    def __init__(self, data_channel):
        super().__init__(MobileNetV3SmallDown16.cfgs, mode='small',
                         scale=0.5, data_channel=data_channel)
        self.set_name(BackboneName.MobileNetV3SmallDown16)

