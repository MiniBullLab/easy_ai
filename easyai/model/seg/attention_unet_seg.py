#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
Attention U-Net: Learning Where to Look for the Pancreas
"""

from easyai.config.name_manager import ModelName
from easyai.config.name_manager import NormalizationType, ActivationType
from easyai.config.name_manager import LayerType
from easyai.config.name_manager import LossName
from easyai.model_block.base_block.seg.unet_blcok import UNetBlockName
from easyai.model_block.base_block.seg.unet_blcok import DoubleConv2d, DownBlock
from easyai.model_block.base_block.seg.unet_blcok import AttentionUpBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.AttentionUnetSeg)
class AttentionUnetSeg(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.AttentionUnetSeg)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        layer1 = DoubleConv2d(in_channels=self.data_channel,
                              out_channels=64,
                              bn_name=self.bn_name,
                              activation_name=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, 64)

        self.down_layers()
        self.up_layers()

        conv = nn.Conv2d(64, self.class_number, kernel_size=1)
        self.add_block_list(LayerType.Convolutional, conv, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {"type": LossName.BinaryCrossEntropy2dLoss}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def down_layers(self):
        down1 = DownBlock(in_channels=64, out_channels=128,
                          bn_name=self.bn_name,
                          activation_name=self.activation_name)
        self.add_block_list(down1.get_name(), down1, 128)

        down2 = DownBlock(in_channels=128, out_channels=256,
                          bn_name=self.bn_name,
                          activation_name=self.activation_name)
        self.add_block_list(down2.get_name(), down2, 256)

        down3 = DownBlock(in_channels=256, out_channels=512,
                          bn_name=self.bn_name,
                          activation_name=self.activation_name)
        self.add_block_list(down3.get_name(), down3, 512)

        down4 = DownBlock(in_channels=512, out_channels=1024,
                          bn_name=self.bn_name,
                          activation_name=self.activation_name)
        self.add_block_list(down4.get_name(), down4, 1024)

    def up_layers(self):
        up1 = AttentionUpBlock(in_channels=1024, out_channels=512,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up1.get_name(), up1, 1024)

        conv1 = DoubleConv2d(in_channels=1024, out_channels=512,
                             bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 512)

        up2 = AttentionUpBlock(in_channels=512, out_channels=256,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up2.get_name(), up2, 512)

        conv2 = DoubleConv2d(in_channels=512, out_channels=256,
                             bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(conv2.get_name(), conv2, 256)

        up3 = AttentionUpBlock(in_channels=256, out_channels=128,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up3.get_name(), up3, 256)

        conv3 = DoubleConv2d(in_channels=256, out_channels=128,
                             bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(conv3.get_name(), conv3, 128)

        up4 = AttentionUpBlock(in_channels=128, out_channels=64,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up4.get_name(), up4, 128)

        conv4 = DoubleConv2d(in_channels=128, out_channels=64,
                             bn_name=self.bn_name,
                             activation_name=self.activation_name)
        self.add_block_list(conv4.get_name(), conv4, 64)

    def forward(self, x):
        layer_outputs = []
        output = []
        index = 3
        for key, block in self._modules.items():
            if UNetBlockName.AttentionUpBlock in key:
                x = block(layer_outputs[-1], layer_outputs[index])
                index -= 1
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            layer_outputs.append(x)
        return output

