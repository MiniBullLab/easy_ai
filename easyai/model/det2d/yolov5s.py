#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.pooling_layer import SpatialPyramidPooling
from easyai.model_block.base_block.common.upsample_layer import Upsample
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.base_block.cls.yolov5_block import FocusBlock, BottleNeck, C3Block
from easyai.model.utility.base_det_model import *
from easyai.model.utility.model_registry import REGISTERED_DET2D_MODEL

class YoloV5sDet2d(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.YoloV5sDet2d)
        self.seq0_Focus = FocusBlock(in_channels=3, out_channels=32, kernel_size=1)
        self.seq1_Conv = ConvBNActivationBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.seq2_C3 = C3Block(64, 64, 1)
        self.seq3_Conv = ConvBNActivationBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.seq4_C3 = C3Block(128, 128, 3)
        self.seq5_Conv = ConvBNActivationBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.seq6_C3 = C3Block(256, 256, 3)
        self.seq7_Conv = ConvBNActivationBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.seq8_SPP = SpatialPyramidPooling(512, 512, [5, 9, 13])
        self.seq9_C3 = C3Block(512, 512, 1, False)
        self.seq10_Conv = ConvBNActivationBlock(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.seq13_C3 = C3Block(512, 256, 1, False)
        self.seq14_Conv = ConvBNActivationBlock(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.seq17_C3 = C3Block(256, 128, 1, False)
        self.seq18_Conv = ConvBNActivationBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2)
        self.seq20_C3 = C3Block(256, 256, 1, False)
        self.seq21_Conv = ConvBNActivationBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.seq23_C3 = C3Block(512, 512, 1, False)
    def forward(self, x):
        x = self.seq0_Focus(x)
        x = self.seq1_Conv(x)
        x = self.seq2_C3(x)
        x = self.seq3_Conv(x)
        xRt0 = self.seq4_C3(x)
        x = self.seq5_Conv(xRt0)
        xRt1 = self.seq6_C3(x)
        x = self.seq7_Conv(xRt1)
        x = self.seq8_SPP(x)
        x = self.seq9_C3(x)
        xRt2 = self.seq10_Conv(x)
        route = F.interpolate(xRt2, size=(int(xRt2.shape[2] * 2), int(xRt2.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, xRt1], dim=1)
        x = self.seq13_C3(x)
        xRt3 = self.seq14_Conv(x)
        route = F.interpolate(xRt3, size=(int(xRt3.shape[2] * 2), int(xRt3.shape[3] * 2)), mode='nearest')
        x = torch.cat([route, xRt0], dim=1)
        out0 = self.seq17_C3(x)
        x = self.seq18_Conv(out0)
        x = torch.cat([x, xRt3], dim=1)
        out1 = self.seq20_C3(x)
        x = self.seq21_Conv(out1)
        x = torch.cat([x, xRt2], dim=1)
        out2 = self.seq23_C3(x)
        return out0, out1, out2

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output