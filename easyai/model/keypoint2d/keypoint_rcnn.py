#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.fpn_block import FPNBlock
from easyai.model.model_block.head.det2d.rpn_head import MultiRPNHead, HeadType
from easyai.model.model_block.head.det2d.roi_box_head import MultiROIBoxHead
from easyai.model.model_block.head.keypoint2d.roi_keypoint_head import MultiROIKeypointHead
from easyai.loss.det2d.utility.rpn_postprocess import RPNPostProcessor
from easyai.model.utility.base_pose_model import *


class KeyPointRCNN(BasePoseModel):

    def __init__(self, data_channel=3, keypoints_number=17):
        super().__init__(data_channel, keypoints_number)
        self.set_name(ModelName.KeyPointRCNN)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.feature_out_channels = 256
        self.anchor_number = 3

        self.model_args['type'] = BackboneName.ResNet50

        self.rpn_postprocess = RPNPostProcessor((640, 640))
        self.nms_thresh = 0.7

        self.rpn_loss_config = {"type": LossName.RPNLoss,
                                "input_size": "640,640",
                                "anchor_sizes": "32,64,128,256,512",
                                "aspect_ratios": "0.5,1.0,2.0",
                                "anchor_strides": "4,8,16,32,64",
                                "fg_iou_threshold": 0.7,
                                "bg_iou_threshold": 0.3,
                                "per_image_sample": 256,
                                "positive_fraction": 0.5}

        self.box_loss_config = {"type": LossName.FastRCNNLoss,
                                "fg_iou_threshold": 0.5,
                                "bg_iou_threshold": 0.5,
                                "per_image_sample": 512,
                                "positive_fraction": 0.25}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        self.create_loss_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        down_layers = [4, 8, 14, 17]
        down_layer_outputs = [self.block_out_channels[i] if i < 0 else base_out_channels[i]
                              for i in down_layers]
        temp_str = ",".join('%s' % index for index in down_layers)
        fpn_layer = FPNBlock(temp_str, down_layer_outputs, self.feature_out_channels)
        self.add_block_list(fpn_layer.get_name(), fpn_layer, 256)

        head_layer1 = MultiRPNHead(self.feature_out_channels, self.anchor_number,
                                   activation_name=self.activation_name)
        self.add_block_list(head_layer1.get_name(), head_layer1, self.anchor_number)

        rpn_loss = self.loss_factory.get_loss(self.rpn_loss_config)
        self.add_block_list(rpn_loss.get_name(), rpn_loss, self.block_out_channels[-1])
        self.lossList.append(rpn_loss)

        head_layer2 = MultiROIBoxHead(self.feature_out_channels, 1024,
                                      class_number=2,
                                      pool_resolution=7,
                                      pool_scales=(1/4, 1/8, 1/16, 1/32),
                                      pool_sampling_ratio=2,
                                      activation_name=self.activation_name)
        self.add_block_list(head_layer2.get_name(), head_layer2, 2)

        box_loss = self.loss_factory.get_loss(self.box_loss_config)
        self.add_block_list(box_loss.get_name(), box_loss, 1024)
        self.lossList.append(box_loss)

        head_layer3 = MultiROIKeypointHead(self.feature_out_channels, self.keypoints_number,
                                           pool_resolution=14,
                                           pool_scales=(1 / 4, 1 / 8, 1 / 16, 1 / 32),
                                           pool_sampling_ratio=2,
                                           activation_name=self.activation_name)
        self.add_block_list(head_layer3.get_name(), head_layer3, self.keypoints_number)

    def create_loss_list(self, input_dict=None):
        self.lossList = []

    def build_rpn_proposals(self, fpn_output, rpn_output, targets):
        anchors, inside_anchors = self.lossList[0].build_anchors(fpn_output)
        with torch.no_grad():
            objectness = []
            box_regression = []
            num_levels = len(fpn_output)
            for index in range(0, num_levels):
                feature_index = index * 2
                objectness.append(rpn_output[feature_index])
                box_regression.append(rpn_output[feature_index + 1])
            predict_boxes = self.rpn_postprocess(anchors, objectness, box_regression)
            sampled_boxes = []
            for level_box_list in predict_boxes:
                result = []
                for temp_box_list in level_box_list:
                    temp_box_list = self.rpn_postprocess.clip_to_image(temp_box_list)
                    temp_box_list = self.rpn_postprocess.remove_small_boxes(temp_box_list, 0)
                    temp_box_list = self.rpn_postprocess.box_list_nms(temp_box_list, self.nms_thresh)
                    result.append(temp_box_list)
                sampled_boxes.append(sampled_boxes)

            box_list = list(zip(*sampled_boxes))
            box_list = [torch.cat(boxes, dim=0) for boxes in box_list]
            if num_levels > 1:
                box_list = self.rpn_postprocess.select_over_all_levels(box_list, True)
            proposals = self.rpn_postprocess.add_gt_proposals(box_list, targets)
        return proposals

    def build_box_proposals(self, proposals, targets):
        if self.training:
            with torch.no_grad():
                proposals = self.self.lossList[1].sample_proposals(proposals, targets)
        return proposals

    def build_keypoint_proposals(self, proposals, targets):
        if self.training:
            with torch.no_grad():
                proposals = self.self.lossList[2].sample_proposals(proposals, targets)
        return proposals

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        multi_output = []
        fpn_output = []
        proposals = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.FPNBlock in key:
                x = block(layer_outputs, base_outputs)
                fpn_output = x
            elif HeadType.MultiRPNHead in key:
                x = block(x)
                multi_output.clear()
                multi_output.extend(fpn_output)
                multi_output.extend(x)
                proposals = self.build_rpn_proposals(fpn_output, x, None)
            elif HeadType.MultiROIBoxHead in key:
                x = block(fpn_output, proposals)
                multi_output.clear()
                multi_output.extend(x)
            elif HeadType.MultiROIKeypointHead in key:
                x = block(fpn_output, proposals)
            elif self.loss_factory.has_loss(key):
                temp_output = self.loss_factory.get_loss_input(key, x, multi_output)
                output.extend(temp_output)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output

