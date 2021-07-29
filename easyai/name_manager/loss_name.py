#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


class LossName():

    # common
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
    RefineMultiBoxLoss = "RefineMultiBoxLoss"
    RPNLoss = "RPNLoss"
    FastRCNNLoss = "FastRCNNLoss"

    # det3d

    # seg
    EncNetLoss = "encNetLoss"
    MixCrossEntropy2dLoss = "MixCE2dLoss"
    MixBinaryCrossEntropy2dLoss = "MixBCELoss"
    DBLoss = "dbLoss"

    # keypoint2d2d
    Keypoint2dRegionLoss = "Keypoint2dRegionLoss"
    Keypoint2dRCNNLoss = "Keypoint2dRCNNLoss"
    DSNTLoss = "DSNTLoss"
    JointsMSELoss = "JointsMSELoss"
    WingLoss = "WingLoss"
    MouthEyeFrontDisLoss = "MouthEyeFrontDisLoss"
    FaceLandmarkLoss = "FaceLandmarkLoss"

    # gen_image
    MNISTDiscriminatorLoss = "MNISTDiscriminatorLoss"
    MNISTGeneratorLoss = "MNISTGeneratorLoss"

    GANomalyGeneratorLoss = "GANomalyGeneratorLoss"
    GANomalyDiscriminatorLoss = "GANomalyDiscriminatorLoss"

    # rnn
    CTCLoss = "CTCLoss"
    TransformerLoss = "TransformerLoss"
    AggregationCrossEntropyLoss = "ACELoss"
    ACELabelSmoothingLoss = "ACELabelSmoothingLoss"

    # pc
    PointNetClsLoss = "pointClsNetLoss"
    PointNetSegLoss = "pointSegNetLoss"
