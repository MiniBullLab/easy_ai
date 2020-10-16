#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""DeepLabV3Plus
    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
"""

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossType
from easyai.loss.cls.ce2d_loss import CrossEntropy2d
from easyai.model.base_block.utility.upsample_layer import Upsample
from easyai.model.base_block.utility.utility_layer import RouteLayer
from easyai.model.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.base_block.utility.separable_conv_block import SeparableConv2dBNActivation
from easyai.model.base_block.cls.deeplab_block import ASPP
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.DeepLabV3Plus)
class DeepLabV3Plus(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=2):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.DeepLabV3Plus)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = BackboneName.Xception65
        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        c1_channels = 256
        c4_channels = 2048

        backbone = self.factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        aspp = ASPP(c4_channels, 256, bn_name=self.bn_name,
                    activation_name=self.activation_name)
        out_channel = 256
        self.add_block_list(aspp.get_name(), aspp, out_channel)

        up1 = Upsample(scale_factor=4, mode='bilinear')
        self.add_block_list(up1.get_name(), up1, self.block_out_channels[-1])

        layer1 = RouteLayer('3')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        conv1 = ConvBNActivationBlock(in_channels=c1_channels,
                                      out_channels=48,
                                      kernel_size=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 48)

        layer2 = RouteLayer('-3, -1')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        block1 = SeparableConv2dBNActivation(output_channel, 256, 3, relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(block1.get_name(), block1, 256)

        block2 = SeparableConv2dBNActivation(256, 256, 3, relu_first=False,
                                             bn_name=self.bn_name,
                                             activation_name=self.activation_name)
        self.add_block_list(block2.get_name(), block2, 256)

        conv2 = nn.Conv2d(256, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional, conv2, self.class_number)

        layer3 = Upsample(scale_factor=4, mode='bilinear')
        self.add_block_list(layer3.get_name(), layer3, self.block_out_channels[-1])

        self.create_loss()

    def create_loss(self, input_dict=None):
        self.lossList = []
        loss = CrossEntropy2d(ignore_index=250)
        self.add_block_list(LossType.CrossEntropy2d, loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.MultiplyLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.AddLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LossType.CrossEntropy2d in key:
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
            print(key, x.shape)
        return output
