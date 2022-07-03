#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.loss.utility.base_loss import *
from easyai.loss.common.common_loss import smooth_l1_loss

from easy_pc.name_manager.pc_loss_name import PCLossName
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_CLS_LOSS
from easy_pc.loss.pc_det3d.utility.anchor_3d_generator import AlignedAnchor3DRangeGenerator
from easy_pc.loss.pc_det3d.utility.delta_xyzwhlr_bbox_coder import DeltaXYZWLHRBBoxCoder
from easy_pc.loss.pc_det3d.utility.max_iou_assigner import MaxIoUAssigner


class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)
            target = target[:, :num_classes]

            loss_cls = self.loss_weight * self.py_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls

    def py_sigmoid_focal_loss(self, pred,
                              target,
                              weight=None,
                              gamma=2.0,
                              alpha=0.25,
                              reduction='mean',
                              avg_factor=None):
        """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Args:
            pred (torch.Tensor): The prediction with shape (N, C), C is the
                number of classes
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) *
                        (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    # For most cases, weight is of shape (num_priors, ),
                    #  which means it does not have the second axis num_class
                    weight = weight.view(-1, 1)
                else:
                    # Sometimes, weight per anchor per class is also needed. e.g.
                    #  in FSAF. But it may be flattened of shape
                    #  (num_priors x num_class, ), while loss is still of shape
                    #  (num_priors, num_class).
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
           assert weight.ndim == loss.ndim
        loss = self.weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    def weight_reduce_loss(self, loss, weight=None, reduction='mean', avg_factor=None):
        """Apply element-wise weight and reduce loss.

        Args:
            loss (Tensor): Element-wise loss.
            weight (Tensor): Element-wise weights.
            reduction (str): Same as built-in losses of PyTorch.
            avg_factor (float): Avarage factor when computing the mean of losses.

        Returns:
            Tensor: Processed loss values.
        """
        # if weight is specified, apply element-wise weight
        if weight is not None:
            loss = loss * weight

        # if avg_factor is not specified, just reduce the loss
        if avg_factor is None:
            loss = self.reduce_loss(loss, reduction)
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')
        return loss

    def reduce_loss(self, loss, reduction):
        """Reduce loss as specified.

        Args:
            loss (Tensor): Elementwise loss tensor.
            reduction (str): Options are "none", "mean" and "sum".

        Return:
            Tensor: Reduced loss tensor.
        """
        reduction_enum = F._Reduction.get_enum(reduction)
        # none: 0, elementwise_mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox



@REGISTERED_PC_CLS_LOSS.register_module(PCLossName.PointPillarsLoss)
class PointPillarsLoss(BaseLoss):

    def __init__(self, class_number,
                anchor_ranges,
                anchor_sizes,
                pos_iou_th,
                neg_iou_th,
                min_pos_iou,
                rotations=(0, 1.57),
                box_code_size=7):
        super().__init__(PCLossName.PointPillarsLoss)
        self.anchor_generator = AlignedAnchor3DRangeGenerator(ranges=anchor_ranges,
                                                              sizes=anchor_sizes,
                                                              rotations=rotations,
                                                              reshape_out=False)
        self.bbox_coder = DeltaXYZWLHRBBoxCoder(box_code_size)
        self.class_number = class_number
        self.box_code_size = box_code_size
        self.assign_per_class = True
        self.sampling = False
        self.bbox_assigner = []
        for pos, neg, min_pos in zip(pos_iou_th, neg_iou_th, min_pos_iou):
            self.bbox_assigner.append(MaxIoUAssigner(pos_iou_thr=pos,
                                                     neg_iou_thr=neg,
                                                     min_pos_iou=min_pos))

    def forward(self, input_list, batch_data=None):
        if batch_data is not None:
            cls_scores = input_list[0]
            bbox_preds = input_list[1]
            dir_cls_preds = input_list[2]
            gt_bboxes = batch_data['gt_bboxes_3d']
            gt_labels = batch_data['gt_labels_3d']
            batch_size, label_channels, height, width = cls_scores.size()
            device = cls_scores.device
            featmap_sizes = [(height, width)]
            multi_level_anchors = self.anchor_generator.grid_anchors(featmap_sizes,
                                                                     device=device)
            anchor_list = [multi_level_anchors for _ in range(batch_size)]
            targets_result = self.anchor_target_3d(anchor_list,
                                                   gt_bboxes,
                                                   gt_labels_list=gt_labels,
                                                   num_classes=self.class_number,
                                                   label_channels=label_channels,
                                                   sampling=self.sampling)
        else:
            loss = None
        return loss

    def anchor_target_3d(self,
                         anchor_list,
                         gt_bboxes_list,
                         gt_labels_list=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            gt_labels_list (list[torch.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of positive anchors and
                number of negative anchors.
        """
        num_imgs = len(anchor_list)

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.view(-1, self.box_code_size).size(0)
                for anchors in anchor_list[0]
            ]
            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        all_labels = []
        all_label_weights = []
        all_bbox_targets = []
        all_bbox_weights = []
        all_dir_targets = []
        all_dir_weights = []
        pos_inds_list = []
        neg_inds_list = []
        for anchor, gt_bboxes, gt_lables in zip(anchor_list,
                                                gt_bboxes_list,
                                                gt_labels_list):
            total_labels, total_label_weights, total_bbox_targets, \
             total_bbox_weights, total_dir_targets, total_dir_weights, \
              total_pos_inds, total_neg_inds = self.anchor_target_3d_single(anchor,
                                                                            gt_bboxes,
                                                                            None,
                                                                            gt_lables,
                                                                            num_classes=num_classes,
                                                                            sampling=sampling)
            all_labels.append(total_labels)
            all_label_weights.append(total_label_weights)
            all_bbox_targets.append(total_bbox_targets)
            all_bbox_weights.append(total_bbox_weights)
            all_dir_targets.append(total_dir_targets)
            all_dir_weights.append(total_dir_weights)
            pos_inds_list.append(total_pos_inds)
            neg_inds_list.append(total_neg_inds)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        if isinstance(self.bbox_assigner,
                      list) and (not isinstance(anchors, list)):
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
            rot_angles = anchors.size(-2)
            assert len(self.bbox_assigner) == anchors.size(-3)
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(
                    -1, self.box_code_size)
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls],
                        num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels.reshape(feat_size, 1, rot_angles))
                total_label_weights.append(
                    label_weights.reshape(feat_size, 1, rot_angles))
                total_bbox_targets.append(
                    bbox_targets.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                total_dir_targets.append(
                    dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(
                    dir_weights.reshape(feat_size, 1, rot_angles))
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
            total_label_weights = torch.cat(
                total_label_weights, dim=-2).reshape(-1)
            total_bbox_targets = torch.cat(
                total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
            total_bbox_weights = torch.cat(
                total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
            total_dir_targets = torch.cat(
                total_dir_targets, dim=-2).reshape(-1)
            total_dir_weights = torch.cat(
                total_dir_weights, dim=-2).reshape(-1)
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(
                anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls],
                        num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(
                    bbox_targets.reshape(-1, anchors[i].size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(-1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=0)
            total_label_weights = torch.cat(total_label_weights, dim=0)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=0)
            total_bbox_weights = torch.cat(total_bbox_weights, dim=0)
            total_dir_targets = torch.cat(total_dir_targets, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_pos_inds = torch.cat(total_pos_inds, dim=0)
            total_neg_inds = torch.cat(total_neg_inds, dim=0)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner,
                                                      anchors,
                                                      gt_bboxes,
                                                      gt_bboxes_ignore,
                                                      gt_labels,
                                                      num_classes,
                                                      sampling)

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long)
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_bboxes) > 0:
            if not isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.tensor.to(anchors.device)
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            sampling_result = self.bbox_sampler.sample(assign_result, anchors,
                                                       gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0,
                as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) == 0,
                as_tuple=False).squeeze(-1).unique()

        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                self.dir_offset,
                self.dir_limit_offset,
                one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, pos_inds, neg_inds)
