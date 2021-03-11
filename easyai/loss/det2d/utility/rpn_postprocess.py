#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


import torch
from easyai.loss.det2d.utility.box_coder import BoxCoder
from easyai.base_algorithm.nms import nms as _box_nms


class RPNPostProcessor(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.pre_nms_top_n = 2000
        self.post_nms_top_n = 2000
        self.fpn_post_nms_top_n = 2000

    def reshape_outputs(self, box_cls, box_regression):
        result_box_cls = []
        result_box_regression = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for box_cls_per_level, box_regression_per_level in zip(
                box_cls, box_regression
        ):
            N, AxC, H, W = box_cls_per_level.shape
            Ax4 = box_regression_per_level.shape[1]
            A = Ax4 // 4
            C = AxC // A

            box_cls_per_level = box_cls_per_level.view(N, -1, C, H, W)
            box_cls_per_level = box_cls_per_level.permute(0, 3, 4, 1, 2)
            box_cls_per_level = box_cls_per_level.reshape(N, -1, C)

            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)

            result_box_cls.append(box_cls_per_level)
            result_box_regression.append(box_regression_per_level)
        return result_box_cls, result_box_regression

    def process_for_single_feature_map(self, anchors, objectness, box_regression):
        device = objectness.device
        N, num_anchors, C = objectness.shape
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        concat_anchors = torch.cat(anchors, dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4)
        )

        proposals = proposals.view(N, -1, 4)

        result = torch.cat([proposals, objectness], dim=2)
        return result

    def clip_to_image(self, box_list, remove_empty=False):
        TO_REMOVE = 1
        box_list[:, 0].clamp_(min=0, max=self.input_size[0] - TO_REMOVE)
        box_list[:, 1].clamp_(min=0, max=self.input_size[1] - TO_REMOVE)
        box_list[:, 2].clamp_(min=0, max=self.input_size[0] - TO_REMOVE)
        box_list[:, 3].clamp_(min=0, max=self.input_size[1] - TO_REMOVE)
        if remove_empty:
            box = box_list
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return box_list[keep]
        return box_list

    def remove_small_boxes(self, box_list, min_size):
        TO_REMOVE = 1  # TODO remove
        widths = box_list[:, 2] - box_list[:, 0] + TO_REMOVE
        heights = box_list[:, 3] - box_list[:, 1] + TO_REMOVE
        keep = ((widths >= min_size) & (heights >= min_size)).nonzero().squeeze(1)
        return box_list[keep]

    def box_list_nms(self, box_list, nms_thresh):
        """
        Performs non-maximum suppression on a boxlist, with scores specified
        in a boxlist field via score_field.

        Arguments:
            box_list(tensor)
            nms_thresh (float)
            max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        """
        if nms_thresh <= 0:
            return box_list
        boxes = box_list[:, :4]
        score = box_list[:, 4]
        score = score.squeeze()
        keep = _box_nms(boxes, score, nms_thresh)
        if self.post_nms_top_n > 0:
            keep = keep[: self.post_nms_top_n]
        box_list = box_list[keep]
        return box_list

    def select_over_all_levels(self, box_list, is_train=False):
        num_images = len(box_list)
        if is_train:
            objectness = torch.cat(
                [boxes[:, 4] for boxes in box_list], dim=0
            )
            objectness = objectness.squeeze()
            box_sizes = [len(boxes) for boxes in box_list]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                box_list[i] = box_list[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = box_list[i][:, 4]
                objectness = objectness.squeeze()
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
                box_list[i] = box_list[i][inds_sorted]
        return box_list

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: tensor
            targets: list[tensor]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device
        gt_boxes = []
        for target in targets:
            bbox = torch.as_tensor(target, dtype=torch.float32, device=device)
            score = torch.ones((len(target), 1), device=device)
            gt_box = torch.cat([bbox, score], dim=1)
            gt_boxes.append(gt_box)

        proposals = [
            torch.cat([proposal, gt_box], dim=0)
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def forward(self, anchors, objectness, box_regression):
        predict_boxes = []
        anchors = list(zip(*anchors))
        for anchor, cls, box in zip(anchors, objectness, box_regression):
            predict_boxes.append(self.process_for_single_feature_map(anchor, cls, box))
        return predict_boxes
