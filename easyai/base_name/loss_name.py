#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class LossName():

    # utility
    EmptyLoss = "emptyLoss"
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
    RPNLoss = "RPNLoss"
    FastRCNNLoss = "FastRCNNLoss"

    # det3d

    # seg
    EncNetLoss = "encNetLoss"
    MixCrossEntropy2dLoss = "MixCE2dLoss"
    MixBinaryCrossEntropy2dLoss = "MixBCELoss"

    # keypoint2d2d
    Keypoint2dRegionLoss = "Keypoint2dRegionLoss"
    Keypoint2dRCNNLoss = "Keypoint2dRCNNLoss"

    # pose2d
    DSNTLoss = "DSNTLoss"
    JointsMSELoss = "JointsMSELoss"

    # gan
    MNISTDiscriminatorLoss = "MNISTDiscriminatorLoss"
    MNISTGeneratorLoss = "MNISTGeneratorLoss"
