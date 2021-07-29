#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


class PostProcessName():

    # cls
    MaxPostProcess = "MaxPostProcess"
    BinaryPostProcess = "BinaryPostProcess"

    # det2d
    SSDPostProcess = "SSDPostProcess"
    YoloPostProcess = "YoloPostProcess"

    # seg
    MaskPostProcess = "MaskPostProcess"
    SegmentPostProcess = "SegmentPostProcess"

    # gen_image
    MNISTPostProcess = "MNISTPostProcess"

    # keypoint2d
    YoloKeypointPostProcess = "YoloKeypointPostProcess"

    # pose2d
    LandmarkPostProcess = "LandmarkPostProcess"
    HeatmapPostProcess = "HeatmapPostProcess"
    MobilePostProcess = "MobilePostProcess"

    # polygon2d
    DBPostProcess = "DBPostProcess"

    # rec_text
    CTCPostProcess = "CTCPostProcess"
    TransformerPostProcess = "TransformerPostProcess"
    ACEPostProcess = "ACEPostProcess"
