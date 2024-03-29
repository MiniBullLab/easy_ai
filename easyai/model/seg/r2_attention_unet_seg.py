#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""
Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)
"""

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.seg.unet_blcok import UNetBlockName
from easyai.model_block.base_block.seg.unet_blcok import AttentionUpBlock
from easyai.model_block.base_block.seg.unet_blcok import RRCNNBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.R2AttentionUNetSeg)
class R2AttentionUNetSeg(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.R2AttentionUNetSeg)
        self.t = 2
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        RRCNN1 = RRCNNBlock(ch_in=self.data_channel, ch_out=64, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN1.get_name(), RRCNN1, 64)

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
        maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.add_block_list(LayerType.MyMaxPool2d, maxpool1, self.block_out_channels[-1])
        RRCNN2 = RRCNNBlock(ch_in=64, ch_out=128, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN2.get_name(), RRCNN2, 128)

        maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.add_block_list(LayerType.MyMaxPool2d, maxpool2, self.block_out_channels[-1])
        RRCNN3 = RRCNNBlock(ch_in=128, ch_out=256, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN3.get_name(), RRCNN3, 256)

        maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.add_block_list(LayerType.MyMaxPool2d, maxpool3, self.block_out_channels[-1])
        RRCNN4 = RRCNNBlock(ch_in=256, ch_out=512, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN4.get_name(), RRCNN4, 512)

        maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.add_block_list(LayerType.MyMaxPool2d, maxpool4, self.block_out_channels[-1])
        RRCNN5 = RRCNNBlock(ch_in=512, ch_out=1024, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN5.get_name(), RRCNN5, 1024)

    def up_layers(self):
        up1 = AttentionUpBlock(in_channels=1024, out_channels=512,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up1.get_name(), up1, 1024)

        RRCNN1 = RRCNNBlock(ch_in=1024, ch_out=512, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN1.get_name(), RRCNN1, 512)

        up2 = AttentionUpBlock(in_channels=512, out_channels=256,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up2.get_name(), up2, 512)

        RRCNN2 = RRCNNBlock(ch_in=512, ch_out=256, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN2.get_name(), RRCNN2, 256)

        up3 = AttentionUpBlock(in_channels=256, out_channels=128,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up3.get_name(), up3, 256)

        RRCNN3 = RRCNNBlock(ch_in=256, ch_out=128, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN3.get_name(), RRCNN3, 128)

        up4 = AttentionUpBlock(in_channels=128, out_channels=64,
                               bn_name=self.bn_name,
                               activation_name=self.activation_name)
        self.add_block_list(up4.get_name(), up4, 128)

        RRCNN4 = RRCNNBlock(ch_in=128, ch_out=64, t=self.t,
                            bn_name=self.bn_name,
                            activation_name=self.activation_name)
        self.add_block_list(RRCNN4.get_name(), RRCNN4, 64)

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
