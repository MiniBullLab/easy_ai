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

    # det2d
    Region2dLoss = "Region2dLoss"
    YoloV3Loss = "YoloV3Loss"
    MultiBoxLoss = "MultiBoxLoss"
    KeyPoints2dRegionLoss = "KeyPoints2dRegionLoss"

    # det3d

    # seg
    EncNetLoss = "encNetLoss"
    OhemCrossEntropy2d = "OhemCrossEntropy2d"

    # gan
    MNISTDiscriminatorLoss = "MNISTDiscriminatorLoss"
    MNISTGeneratorLoss = "MNISTGeneratorLoss"
