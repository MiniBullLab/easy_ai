#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.matcher import Matcher
from easyai.loss.det2d.utility.select_positive_negative_sampler import SelectPositiveNegativeSampler
from easyai.loss.det2d.utility.box_coder import BoxCoder
from easyai.loss.utility.box2d_process import torch_corners_box2d_ious
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS


@REGISTERED_DET2D_LOSS.register_module(LossName.FastRCNNLoss)
class FastRCNNLoss(BaseLoss):

    def __init__(self, fg_iou_threshold=0.5, bg_iou_threshold=0.5,
                 per_image_sample=512, positive_fraction=0.25):
        super().__init__(LossName.FastRCNNLoss)
        self.matcher = Matcher(fg_iou_threshold, bg_iou_threshold,
                               allow_low_quality_matches=False)

        self.fg_bg_sampler = SelectPositiveNegativeSampler(
            per_image_sample, positive_fraction)

        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        self.labels = None
        self.regression_targets = None

        self.loss_info = {'fastrcnn_box_loss': 0.0, 'fastrcnn_cls_loss': 0.0}

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = torch_corners_box2d_ious(target, proposal)
        matched_idxs = self.matcher(match_quality_matrix)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        return matched_targets, matched_idxs

    def build_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, matched_idxs = self.match_targets_to_proposals(proposals_per_image,
                                                                            targets_per_image)
            labels_per_image = matched_targets[:, 0] + 1
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets[:, 1:], proposals_per_image[:, :4]
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def sample_proposals(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
                proposals (list[Tensor])
                targets (list[Tensor])
        """
        labels, regression_targets = self.build_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image
            label_per_image = labels[img_idx][img_sampled_inds]
            labels[img_idx] = label_per_image
            regression_per_image = regression_targets[img_idx][img_sampled_inds]
            regression_targets[img_idx] = regression_per_image
        self.labels = labels
        self.regression_targets = regression_targets
        return proposals

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (list[Tensor]))
            targets (list[Tensor])

        Returns:
            loss (Tensor)
        """
        class_logits = outputs[0]
        box_regression = outputs[1]
        device = class_logits.device

        if (self.labels is None) or (self.regression_targets is None):
            raise RuntimeError("sample_proposals needs to be called before")

        labels = torch.cat(self.labels, dim=0)
        regression_targets = torch.cat(self.regression_targets, dim=0)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]

        cls_loss = F.cross_entropy(class_logits, labels)

        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = F.smooth_l1_loss(box_regression[sampled_pos_inds_subset[:, None], map_inds],
                                    regression_targets[sampled_pos_inds_subset],
                                    reduction='none')
        box_loss = box_loss.sum() / labels.numel()

        self.loss_info['fastrcnn_box_loss'] = float(box_loss.item())
        self.loss_info['fastrcnn_cls_loss'] = float(cls_loss.item())

        all_loss = box_loss + cls_loss

        return all_loss
