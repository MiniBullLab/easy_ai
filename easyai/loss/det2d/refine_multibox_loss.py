#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.config.name_manager import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.base_multi_loss import BaseMultiLoss
from easyai.loss.utility.box2d_process import torch_rect_box_ious


class RefineMultiBoxLoss(BaseMultiLoss):

    def __init__(self, class_number, iou_threshold, input_size,
                 anchor_counts, aspect_ratios,
                 min_sizes, max_sizes=None,
                 use_arm=False, is_gaussian=False):
        super().__init__(LossName.RefineMultiBoxLoss, class_number, input_size,
                         anchor_counts, aspect_ratios, min_sizes, max_sizes, is_gaussian)
        self.feature_count = len(anchor_counts)
        self.iou_threshold = iou_threshold
        self.use_arm = use_arm
        self.variances = (0.1, 0.2)
        self.giou = False

    def encode(self, matched, priors, variances):
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have matched (based on jaccard overlap) with the prior boxes.
        Args:
            matched: (tensor) Coords of ground truth for each prior in point-form
                Shape: [num_priors, 4].
            priors: (tensor) Prior boxes in center-offset form
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            encoded boxes (tensor), Shape: [num_priors, 4]
        """

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

    def match_s3fd(self, threshold, truths, priors, variances, labels):
        """Match each prior box with the ground truth box of the highest jaccard
        overlap, encode the bounding boxes, then return the matched indices
        corresponding to both confidence and location preds.
        Args:
            threshold: (float) The overlap threshold used when mathing boxes.
            truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
            priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
            variances: (tensor) Variances corresponding to each prior coord,
                Shape: [num_priors, 4].
            labels: (tensor) All the class labels for the image, Shape: [num_obj].
        Return:
            The matched indices corresponding to 1)location and 2)confidence preds.
        """
        # S3FD Threshs 0.5 0.35 0.1 from paper
        # assert isinstance(threshold, (list, tuple)), 'input threshold should be tuple or list'
        # thre1, thre2, thre3 = threshold
        thre2 = threshold
        thre3 = 0.1
        # jaccard index
        overlaps = torch_rect_box_ious(truths, priors)
        # (Bipartite Matching)
        # [1,num_objects] best prior for each ground truth
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

        # ignore hard gt
        '''valid_gt_idx = best_prior_overlap[:, 0] >= thre3
        best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
        best_prior_idx = best_prior_idx[valid_gt_idx, :]
        if best_prior_overlap.shape[0] <= 0:
            #print('??????????????????????????????????')
            loc_t[idx] = 0
            conf_t[idx] = 0
            return'''

        # [1,num_priors] best ground truth for each prior
        best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
        best_truth_idx.squeeze_(0)
        best_truth_overlap.squeeze_(0)
        best_prior_idx.squeeze_(1)
        best_prior_overlap.squeeze_(1)
        best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
        # TODO refactor: index  best_prior_idx with long tensor
        # ensure every gt matches with its prior of max overlap
        for j in range(best_prior_idx.size(0)):
            best_truth_idx[best_prior_idx[j]] = j
        matches = truths[best_truth_idx]  # Shape: [num_priors,4]
        conf = labels[best_truth_idx]  # Shape: [num_priors]
        conf[best_truth_overlap < thre2] = 0  # label as background
        # print(torch.sum(conf == 1).item())

        # S3FD Scale Compensation anchor matching strategy
        num_truths = truths.size(0)
        N = int(best_truth_idx[best_truth_overlap > thre2].size(0) / num_truths)
        # print(best_truth_idx[best_truth_overlap > thre2], N)
        for i in range(num_truths):
            i_num_anchors = torch.sum((best_truth_overlap > thre2).add(best_truth_idx == i) == 2).item()
            # print(i_num_anchors)
            if i_num_anchors >= N:
                continue

            best_truth_overlap_t = best_truth_overlap.clone()
            thre_idx = (best_truth_overlap_t >= thre3).add(best_truth_idx == i) == 2
            best_truth_overlap_t[1 - thre_idx] = 0
            sort_value, sort_idx = best_truth_overlap_t.sort(descending=True)
            M = torch.sum(sort_value.ne(0)).item()
            # print(sort_value, sort_idx, M)
            if M <= N:
                conf[sort_idx[0:M]] = 1
            else:
                conf[sort_idx[0:N]] = 1
            del best_truth_overlap_t, thre_idx, sort_value, sort_idx
        # print(torch.sum(conf == 1).item())

        # S3FD Scale Compensation anchor matching strategy
        '''N = torch.sum(best_truth_overlap > thre2).item()
        best_truth_overlap_t = best_truth_overlap.clone()
        thre_idx = (best_truth_overlap_t >= thre3).eq(best_truth_overlap_t <= thre2)
        best_truth_overlap_t[1 - thre_idx] = 0
        sort_value, sort_idx = best_truth_overlap_t.sort(descending=True)
        M = torch.sum(sort_value.ne(0)).item()
        if M <= N:
            conf[sort_idx[0:M]] = 1
        else:
            conf[sort_idx[0:N]] = 1'''
        if not self.giou:
            # [num_priors,4] encoded offsets to learn
            loc = self.encode(matches, priors, variances)
            return loc, conf
        else:
            return matches, conf  # [num_priors] top class label for each prior

    def build_targets(self, gt_targets, prior_boxes):
        N = len(gt_targets)
        target_boxes = []
        target_labels = []
        for index in range(N):
            gt_data = gt_targets[index]
            if gt_data is None or len(gt_data) == 0:
                loc = torch.zeros_like(prior_boxes)
                cls_conf = torch.zeros((prior_boxes.size(0),)).long()
            else:
                gt_boxes = gt_data[:, 0:]
                gt_labels = gt_data[:, 0]
                if not self.use_arm:
                    gt_labels = gt_labels >= 0
                    loc, cls_conf = self.match_s3fd(self.iou_threshold,
                                                    gt_boxes,
                                                    prior_boxes,
                                                    self.variances,
                                                    gt_labels)
                else:
                    gt_labels = gt_labels + 1
            target_boxes.append(loc)
            target_labels.append(cls_conf)
        return torch.stack(target_boxes, 0), torch.stack(target_labels, 0)

    def forward(self, output_list, targets=None):
        if self.use_arm:
            output_count = len(output_list) // 2
            arm_output_list = output_list[:output_count]
            arm_loc_preds, arm_cls_preds, feature_sizes = self.reshape_box_outputs(arm_output_list)
            loc_preds, cls_preds, _ = self.reshape_box_outputs(output_list[output_count:])
            if self.is_gaussian:
                # If use gaussian, split the output with xywh and its uncertainties
                xq_split = torch.split(loc_preds, 4, 2)
                loc_preds = xq_split[0]
                sigma_xywh = xq_split[1]
                sigma_xywh = sigma_xywh.sigmoid()
        else:
            loc_preds, cls_preds, feature_sizes = self.reshape_box_outputs(output_list)
        anchor_boxes = self.priorbox(self.input_size, feature_sizes)
        loc_targets, cls_targets = self.build_targets(targets, anchor_boxes)
        # wrap targets
        loc_targets = Variable(loc_targets, requires_grad=False)
        cls_targets = Variable(cls_targets, requires_grad=False)
        if self.use_arm:
            P = F.softmax(arm_cls_preds, 2)
            arm_conf_data_temp, _ = P[:, :, 1:].max(dim=2)
            object_score_index = arm_conf_data_temp <= 0.01
            pos = cls_targets > 0
            pos[object_score_index.detach()] = 0
        else:
            pos = cls_targets > 0
        num_pos = pos.sum(1, keepdim=True)
