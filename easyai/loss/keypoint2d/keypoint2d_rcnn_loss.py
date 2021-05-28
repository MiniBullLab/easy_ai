#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.matcher import Matcher
from easyai.loss.det2d.utility.select_positive_negative_sampler import SelectPositiveNegativeSampler
from easyai.loss.utility.box2d_process import torch_corners_box2d_ious


class Keypoint2dRCNNLoss(BaseLoss):

    def __init__(self, fg_iou_threshold=0.5, bg_iou_threshold=0.5,
                 per_image_sample=512, positive_fraction=0.25,
                 discretization_size=56):
        super().__init__(LossName.Keypoint2dRCNNLoss)
        self.matcher = Matcher(fg_iou_threshold, bg_iou_threshold,
                               allow_low_quality_matches=False)

        self.fg_bg_sampler = SelectPositiveNegativeSampler(
            per_image_sample, positive_fraction)

        self.discretization_size = discretization_size

        self.labels = None
        self.keypoints = None

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = torch_corners_box2d_ious(target[:, 1:5], proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    def _within_box(self, points, boxes):
        """Validate which keypoints are contained inside a given box.
        points: NxKx2
        boxes: Nx4
        output: NxK
        """
        x_within = (points[..., 0] >= boxes[:, 0, None]) & (
                points[..., 0] <= boxes[:, 2, None]
        )
        y_within = (points[..., 1] >= boxes[:, 1, None]) & (
                points[..., 1] <= boxes[:, 3, None]
        )
        return x_within & y_within

    def build_targets(self, proposals, targets):
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, matched_idxs = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )

            labels_per_image = matched_targets[:, 0] + 1
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # # this can probably be removed, but is left here for clarity
            # # and completeness
            # # TODO check if this is the right one, as BELOW_THRESHOLD
            # neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            # labels_per_image[neg_inds] = 0
            #
            # keypoints_per_image = matched_targets[:, 5:]
            # within_box = _within_box(
            #     keypoints_per_image.keypoints, matched_targets.bbox
            # )
            # vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            # is_visible = (within_box & vis_kp).sum(1) > 0
            #
            # labels_per_image[~is_visible] = -1
            #
            # labels.append(labels_per_image)
            # keypoints.append(keypoints_per_image)

        return labels, keypoints

    def sample_proposals(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
                proposals (list[Tensor])
                targets (list[Tensor])
        """
        pass

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (list[Tensor]))
            targets (list[Tensor])

        Returns:
            loss (Tensor)
        """
        pass
