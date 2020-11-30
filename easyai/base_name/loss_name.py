#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class LossName():

    # utility
    MeanSquaredErrorLoss = "mseLoss"

    # cls
    CrossEntropy2dLoss = "crossEntropy2dLoss"
    BinaryCrossEntropy2dLoss = "bceLoss"
    LabelSmoothCE2dLoss = "LabelSmoothCE2dLoss"
    FocalLoss = "FocalLoss"
    FocalBinaryLoss = "FocalBinaryLoss"
    OhemCrossEntropy2dLoss = "OhemCE2dLoss"
    OhemBinaryCrossEntropy2dLoss = "OhemBCELoss"

    # det2d
    Region2dLoss = "Region2dLoss"
    YoloV3Loss = "YoloV3Loss"
    MultiBoxLoss = "MultiBoxLoss"
    KeyPoints2dRegionLoss = "KeyPoints2dRegionLoss"

    # det3d

    # seg
    EncNetLoss = "encNetLoss"
    MixCrossEntropy2dLoss = "MixCE2dLoss"
    MixBinaryCrossEntropy2dLoss = "MixBCELoss"

    # gan
    MNISTDiscriminatorLoss = "MNISTDiscriminatorLoss"
    MNISTGeneratorLoss = "MNISTGeneratorLoss"
