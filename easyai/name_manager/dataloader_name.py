#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


class DatasetName():

    ClassifyDataSet = "ClassifyDataSet"
    Det2dDataset = "Det2dDataset"
    Det2dSegDataset = "Det2dSegDataset"
    GenImageDataset = "GenImageDataset"
    KeyPoint2dDataset = "KeyPoint2dDataset"
    LandmarkDataset = "LandmarkDataset"
    OneClassDataset = "OneClassDataset"
    Pose2dDataset = "Pose2dDataset"
    SegmentDataset = "SegmentDataset"
    RecTextDataSet = "RecTextDataSet"
    RecTextOCRDataSet = "RecTextOCRDataSet"
    SuperResolutionDataset = "SuperResolutionDataset"
    DetOCRDataSet = "DetOCRDataSet"

    ClassifyPointCloudDataSet = "ClassifyPointCloudDataSet"


class DatasetCollateName():

    ClassifyDataSetCollate = "ClassifyDataSetCollate"
    Det2dDataSetCollate = "Det2dDataSetCollate"
    KeyPoint2dDataSetCollate = "KeyPoint2dDataSetCollate"
    Pose2dDataSetCollate = "Pose2dDataSetCollate"
    RecTextDataSetCollate = "RecTextDataSetCollate"
    SuperResolutionDataSetCollate = "SuperResolutionDataSetCollate"
    SegmentDataSetCollate = "SegmentDataSetCollate"
    DetOCRDataSetCollate = "DetOCRDataSetCollate"

    ClassifyPointCloudDataSetCollate = "ClassifyPointCloudDataSetCollate"


class DataloaderName():

    Det2dTrainDataloader = "Det2dTrainDataloader"
    Det2dSegTrainDataloader = "Det2dSegTrainDataloader"
    DataLoader = "DataLoader"


class DataTansformsName():

    ImageWidthSlide = "ImageWidthSlide"
