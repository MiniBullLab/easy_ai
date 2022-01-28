#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.utility.base_block import *
import warnings


class Yolov5BlockName():

    FocusBlock = "focusBlock"
    BottleNeck = "bottleNeck"
    C3Block = "C3Block"
    SpatialPyramidPoolingWithConv = "SPPBlockWithConv"
    SPPFBlock = "SPPFBlock"


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class FocusBlock(BaseBlock):
    # Focus wh information into c-space
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.Swish):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(Yolov5BlockName.FocusBlock)
        self.conv = ConvBNActivationBlock(in_channels=in_channels * 4,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          groups=groups,
                                          bnName=bnName,
                                          activationName=activationName)
        self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))
        return self.conv(self.contract(x))


class BottleNeck(BaseBlock):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.Swish,
                 expansion=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(Yolov5BlockName.BottleNeck)
        channels_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=channels_,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)
        self.conv2 = ConvBNActivationBlock(in_channels=channels_,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           groups=groups,
                                           bnName=bnName,
                                           activationName=activationName)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C3Block(BaseBlock):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_channels, out_channels, number=1, shortcut=True, groups=1,
                 bnName=NormalizationType.BatchNormalize2d,
                 activationName=ActivationType.Swish,
                 expansion=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(Yolov5BlockName.C3Block)
        channels_ = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=channels_,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)
        self.conv2 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=channels_,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)
        self.conv3 = ConvBNActivationBlock(in_channels=2 * channels_,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)  # act=FReLU(c2)
        self.m_conv = nn.Sequential(*[BottleNeck(channels_, channels_, shortcut=shortcut, groups=groups, expansion=1.0)
                                 for _ in range(number)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.m_conv(self.conv1(x)), self.conv2(x)), dim=1))


# spp block for yolov5(not the same as SpatialPyramidPooling)
class SpatialPyramidPoolingWithConv(BaseBlock):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, in_channels, out_channels, pool_sizes=(5, 9, 13),
                 bnName=NormalizationType.BatchNormalize2d, activationName=ActivationType.Swish):
        super().__init__(Yolov5BlockName.SpatialPyramidPoolingWithConv)
        channels_ = in_channels // 2  # hidden channels
        self.conv1 = ConvBNActivationBlock(in_channels=in_channels,
                                           out_channels=channels_,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)
        self.conv2 = ConvBNActivationBlock(in_channels=channels_ * (len(pool_sizes) + 1),
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bnName=bnName,
                                           activationName=activationName)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in pool_sizes])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.maxpools], 1))


class SPPFBlock(BaseBlock):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5,
                 bn_name=NormalizationType.BatchNormalize2d,
                 activation_name=ActivationType.Swish):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__(Yolov5BlockName.SPPFBlock)
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBNActivationBlock(c1, c_, 1, 1,
                                         bnName=bn_name,
                                         activationName=activation_name)
        self.cv2 = ConvBNActivationBlock(c_ * 4, c2, 1, 1,
                                         bnName=bn_name,
                                         activationName=activation_name)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
