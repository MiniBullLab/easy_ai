#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.anchor_generator import AnchorGenerator
from easyai.loss.det2d.utility.matcher import Matcher
from easyai.loss.det2d.utility.select_positive_negative_sampler import SelectPositiveNegativeSampler
from easyai.loss.det2d.utility.box_coder import BoxCoder
from easyai.loss.det2d.utility.rpn_postprocess import RPNPostProcessor
from easyai.tasks.utility.box2d_process import torch_corners_box2d_ious
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


@REGISTERED_DET2D_LOSS.register_module(LossName.RPNLoss)
class RPNLoss(BaseLoss):

    def __init__(self, input_size, anchor_sizes,
                 aspect_ratios, anchor_strides,
                 fg_iou_threshold=0.5, bg_iou_threshold=0.5,
                 per_image_sample=256, positive_fraction=0.5):
        super().__init__(LossName.RPNLoss)
        if len(anchor_strides) > 1:
            assert len(anchor_strides) == len(
                anchor_sizes
            ), "FPNLoss should have len(ANCHOR_STRIDES) == len(ANCHOR_SIZES)"
        self.input_size = input_size
        self.class_number = 1
        self.multi_feature_count = len(anchor_strides)
        self.anchor_generator = AnchorGenerator(image_size=input_size,
                                                sizes=anchor_sizes,
                                                aspect_ratios=aspect_ratios,
                                                anchor_strides=anchor_strides)

        self.matcher = Matcher(fg_iou_threshold, bg_iou_threshold,
                               allow_low_quality_matches=True)

        self.fg_bg_sampler = SelectPositiveNegativeSampler(
            per_image_sample, positive_fraction)

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.discard_cases = ['not_visibility', 'between_thresholds']

        self.rpn_postprocess = RPNPostProcessor(input_size)

        self.loss_info = {'box_loss': 0.0, 'cls_loss': 0.0}

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = torch_corners_box2d_ious(target, anchor)
        matched_idxs = self.matcher(match_quality_matrix)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    def build_targets(self, anchors, inside_anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets, matched_idxs = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image)

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~inside_anchors] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets, anchors_per_image)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (list[Tensor]))
            targets (list[Tensor])

        Returns:
            loss (Tensor)
        """
        features = outputs[:self.multi_feature_count]
        objectness = []
        box_regression = []
        for index in range(0, self.multi_feature_count):
            feature_index = self.multi_feature_count + index * 2
            objectness.append(outputs[feature_index])
            box_regression.append(outputs[feature_index+1])

        objectness, box_regression = self.rpn_postprocess.reshape_outputs(objectness, box_regression)

        anchors, inside_anchors = self.anchor_generator(features)

        if targets is None:
            proposals = self.rpn_postprocess(anchors, objectness, box_regression)
            return torch.cat(proposals, dim=1)
        else:
            anchors = [torch.cat(anchors_per_image, dim=0) for anchors_per_image in anchors]
            inside_anchors = [torch.cat(inside_per_image, dim=0) for inside_per_image in inside_anchors]

            labels, regression_targets = self.build_targets(anchors, inside_anchors, targets)

            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            objectness = torch.cat(objectness, dim=1).reshape(-1, self.class_number)
            box_regression = torch.cat(box_regression, dim=1).reshape(-1, 4)

            objectness = objectness.squeeze()
            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)

            box_loss = F.smooth_l1_loss(box_regression[sampled_pos_inds],
                                        regression_targets[sampled_pos_inds])
            objectness_loss = F.binary_cross_entropy(
                objectness[sampled_inds], labels[sampled_inds])

            self.loss_info['box_loss'] = float(box_loss.item())
            self.loss_info['cls_loss'] = float(objectness_loss.item())

            all_loss = box_loss + objectness_loss

            return all_loss
