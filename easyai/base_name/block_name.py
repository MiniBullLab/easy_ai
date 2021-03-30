#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class ActivationType():

    Linear = "linear"
    ReLU = "relu"
    PReLU = "prelu"
    ReLU6 = "relu6"
    LeakyReLU = "leaky"
    Sigmoid = "sigmoid"
    Tanh = "tanh"

    Swish = "swish"
    Mish = "mish"

    HardSigmoid = "hard_sigmoid"
    HardSwish = "hard_swish"


class NormalizationType():

    BatchNormalize2d = "bn2d"
    BatchNormalize1d = "bn1d"

    FrozenBatchNorm2d = "frozen_bn2d"

    InstanceNorm2d = "in2d"
    InstanceNorm1d = "in1d"

    EmptyNormalization = "empty"


class LayerType():

    EmptyLayer = "emptyLayer"

    MultiplyLayer = "multiply"
    AddLayer = "add"

    NormalizeLayer = "normalize"
    ActivationLayer = "activation"

    RouteLayer = "route"
    ShortRouteLayer = "shortRoute"
    ShortcutLayer = "shortcut"

    Upsample = "upsample"
    PixelShuffle = "pixelShuffle"

    MyMaxPool2d = "maxpool"
    GlobalAvgPool = "globalavgpool"

    FcLayer = "fcLayer"
    Dropout = "dropout"

    FcLinear = "fcLinear"

    Convolutional1d = "convolutional1d"

    Convolutional = "convolutional"

    ConvTranspose = "deconv"

    MeanLayer = "mean"


class BlockType():

    InputData = "inputData"
    BaseNet = "baseNet"
    Discriminator = "ganDNet"
    Generator = "ganGNet"

    ConvBNBlock1d = "convBN1d"
    ConvBNActivationBlock1d = "convBNActivationBlock1d"

    ConvBNActivationBlock = "convBNActivationBlock"
    BNActivationConvBlock = "bnActivationConvBlock"
    ActivationConvBNBlock = "activationConvBNBlock"

    ConvActivationBlock = "convActivationBlock"

    FcBNActivationBlock = "fcBNActivationBlock"
    FcActivationBlock = "fcActivationBlock"

    InceptionBlock = "inceptionBlock"

    SeperableConv2dBlock = "seperableConv2dBlock"
    DepthwiseConv2dBlock = "depthwiseConv2dBlock"
    SeparableConv2dBNActivation = "separableConv2dBNActivation"
    ShuffleBlock = "shuffleBlock"
    MixConv2dBlock = "mixConv2dBlock"

    ResidualBlock = "residualBlock"
    InvertedResidual = "invertedResidual"

    FPNBlock = "FPNBlock"

    SEBlock = "seBlock"
    SEConvBlock = "seConvBlock"

    SpatialPyramidPooling = "SPPBlock"
    MultiROIPooling = "multiROIPool"

    DenseUpsamplingConvBlock = "ducBlock"
    DeConvBNActivationBlock = "deconvBNActivationBlock"

    # det2d
    Detection2dBlock = "detection2dBlock"

    # gan
    DCEncoder = "DCEncoder"
    DCDecoder = "DCDecoder"


class HeadType():

    MultiRPNHead = "multiRPN"
    MultiROIBoxHead = "multiROIBox"
    MultiROIKeypointHead = "multiROIKeypoint"
    MultiROIMaskHead = "multiROIMask"

