#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.det2d.utility.base_yolo_loss import BaseYoloLoss
from easyai.loss.utility.box2d_process import torch_rect_box_ious
from easyai.loss.utility.loss_registry import REGISTERED_DET2D_LOSS
import math

__all__ = ['YoloV3Loss']


@REGISTERED_DET2D_LOSS.register_module(LossName.YoloV3Loss)
class YoloV3Loss(BaseYoloLoss):

    def __init__(self, class_number, anchor_sizes, anchor_mask, reduction,
                 coord_weight=3.0, noobject_weight=1.0,
                 object_weight=1.0, class_weight=1.0, iou_threshold=0.5):
        super().__init__(LossName.YoloV3Loss, class_number, anchor_sizes, anchor_mask)
        self.reduction = reduction
        self.coord_xy_weight = 2.0 * coord_weight
        self.coord_wh_weight = 3.0 * coord_weight
        self.noobject_weight = noobject_weight
        self.object_weight = object_weight
        self.class_weight = class_weight
        self.iou_threshold = iou_threshold

        self.anchor_sizes = self.anchor_sizes / float(self.reduction)

        # criterion
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')

        self.loss_info = {'object_count': 0, 'object_current': 0,
                          'average_iou': 0, 'recall50': 0, 'recall75': 0,
                          'cls_loss': 0.0, 'obj_loss': 0.0, 'no_obj_loss': 0.0,
                          'coord_xy_loss': 0.0, 'coord_wh_loss': 0.0}

    def build_targets(self, pred_boxes, gt_targets, height, width, device):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        batch_size = len(gt_targets)
        nPixels = height * width

        # Tensors
        object_mask = torch.zeros(batch_size, self.anchor_count, nPixels,
                                  requires_grad=False, dtype=torch.uint8, device=device)
        no_object_mask = torch.ones(batch_size, self.anchor_count, nPixels,
                                    requires_grad=False, dtype=torch.uint8, device=device)
        coord_mask = torch.zeros(batch_size, self.anchor_count, nPixels, 1,
                                 requires_grad=False, device=device)
        cls_mask = torch.zeros(batch_size, self.anchor_count, nPixels,
                               requires_grad=False, dtype=torch.uint8, device=device)
        tcoord = torch.zeros(batch_size, self.anchor_count, nPixels, 4,
                             requires_grad=False, device=device)
        tconf = torch.zeros(batch_size, self.anchor_count, nPixels,
                            requires_grad=False, device=device)
        tcls = torch.zeros(batch_size, self.anchor_count, nPixels,
                           requires_grad=False, device=device)

        recall50 = 0
        recall75 = 0
        object_current = 0
        object_count = 0
        iou_sum = 0
        for b in range(batch_size):
            gt_data = gt_targets[b]
            pred_box = pred_boxes[b]
            if len(gt_data) == 0:  # No gt for this image
                continue
            object_count += len(gt_data)
            anchors = self.scale_anchor()
            gt = self.gt_process.scale_gt_box(gt_data, width, height).to(device)

            # Find best anchor for each gt
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = torch_rect_box_ious(gt_wh, anchors)
            _, best_index = iou_gt_anchors.max(1)

            # Set confidence mask of matching detections to 0
            iou_gt_pred = torch_rect_box_ious(gt, pred_box)
            mask = (iou_gt_pred > self.iou_threshold).sum(0) >= 1
            no_object_mask[b][mask.view_as(no_object_mask[b])] = 0

            # Set masks and target values for each gt
            # time consuming
            for i, anno in enumerate(gt_data):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_index[i]
                if best_n in self.anchor_mask:
                    anchor_index = self.anchor_mask.index(best_n)
                else:
                    continue
                iou = iou_gt_pred[i][anchor_index * nPixels + gj * width + gi]
                # debug information
                object_current += 1
                recall50 += (iou > 0.5).item()
                recall75 += (iou > 0.75).item()
                iou_sum += iou.item()

                object_mask[b][anchor_index][gj * width + gi] = 1
                no_object_mask[b][anchor_index][gj * width + gi] = 0
                coord_mask[b][anchor_index][gj * width + gi][0] = 2 - anno[3] * anno[4] / \
                                                                  (width * height * self.reduction * self.reduction)
                tcoord[b][anchor_index][gj * width + gi][0] = gt[i, 0] - gi
                tcoord[b][anchor_index][gj * width + gi][1] = gt[i, 1] - gj
                tcoord[b][anchor_index][gj * width + gi][2] = math.log(gt[i, 2] / self.anchor_sizes[best_n, 0])
                tcoord[b][anchor_index][gj * width + gi][3] = math.log(gt[i, 3] / self.anchor_sizes[best_n, 1])
                tconf[b][anchor_index][gj * width + gi] = 1
                cls_mask[b][anchor_index][gj * width + gi] = 1
                tcls[b][anchor_index][gj * width + gi] = anno[0]
        # informaion
        if object_current > 0:
            self.loss_info['object_count'] = object_count
            self.loss_info['object_current'] = object_current
            self.loss_info['average_iou'] = iou_sum / object_current
            self.loss_info['recall50'] = recall50 / object_current
            self.loss_info['recall75'] = recall75 / object_current

        return coord_mask, object_mask, no_object_mask, \
               cls_mask, tcoord, tconf, tcls

    def forward(self, outputs, batch_data=None):
        """ Compute Yolo loss.
        """
        # Parameters
        batch_size, C, height, width = outputs.size()
        device = outputs.device
        self.anchor_sizes = self.anchor_sizes.to(device)

        outputs = outputs.view(batch_size, self.anchor_count,
                               5 + self.class_number,
                               height, width)
        outputs = outputs.view(batch_size, self.anchor_count, -1,
                               height * width)

        # Get x,y,w,h,conf,cls
        coord = torch.zeros_like(outputs[:, :, :4, :])
        coord[:, :, :2, :] = outputs[:, :, :2, :].sigmoid()  # tx,ty
        coord[:, :, 2:4, :] = outputs[:, :, 2:4, :]  # tw,th
        conf = outputs[:, :, 4, :].sigmoid()
        conf = conf.view(batch_size, -1, 1)
        cls = outputs[:, :, 5:, :].transpose(2, 3).contiguous().view(batch_size, -1, self.class_number)
        # Create prediction boxes
        pred_boxes = self.decode_predict_box(coord, batch_size, height, width, device)
        pred_boxes = pred_boxes.view(batch_size, -1, 4)

        if batch_data is None:
            pred_boxes *= self.reduction
            cls = F.softmax(cls, 2)
            return torch.cat([pred_boxes, conf, cls], 2)
        else:
            targets = batch_data['label']
            coord_mask, object_mask, no_object_mask, \
            cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, targets, height, width, device)

            # coord
            # 0 = 1 = 2 = 3, only need first two element
            mask_count = object_mask.sum().item()
            coord = coord.transpose(2, 3).contiguous()
            coord_mask = coord_mask.expand_as(tcoord)[:, :, :, :2]
            coord_center, tcoord_center = coord[:, :, :, :2], tcoord[:, :, :, :2]
            coord_wh, tcoord_wh = coord[:, :, :, 2:], tcoord[:, :, :, 2:]

            conf = conf.view(batch_size, self.anchor_count, height * width)
            pos_conf = conf[object_mask].view(-1, 1)
            pos_tconf = tconf[object_mask].view(-1, 1)
            neg_conf = conf[no_object_mask].view(-1, 1)
            neg_tconf = tconf[no_object_mask].view(-1, 1)

            # Compute losses
            # x,y BCELoss; w,h SmoothL1Loss, conf BCELoss, class CELoss
            if mask_count > 0:
                xy_bce_loss = F.binary_cross_entropy(coord_center, tcoord_center,
                                                     reduction='none')
                loss_coord_center = (coord_mask * xy_bce_loss).sum() / mask_count
                wh_l1_loss = F.smooth_l1_loss(coord_wh, tcoord_wh, reduction='none')
                loss_coord_wh = (coord_mask * wh_l1_loss).sum()

                loss_conf_pos = self.object_weight * self.bce_loss(pos_conf, pos_tconf)
            else:
                loss_coord_center = torch.tensor(0.0, device=device)
                loss_coord_wh = torch.tensor(0.0, device=device)
                loss_conf_pos = torch.tensor(0.0, device=device)

            loss_coord = self.coord_xy_weight * loss_coord_center + \
                         self.coord_wh_weight * loss_coord_wh

            loss_conf_neg = self.noobject_weight * self.bce_loss(neg_conf, neg_tconf)
            loss_conf = loss_conf_pos + loss_conf_neg

            if self.class_number > 1 and cls_mask.sum().item() > 0:
                cls = cls.view(-1, self.class_number)
                tcls = tcls[cls_mask].view(-1).long()
                cls_mask = cls_mask.view(-1, 1).repeat(1, self.class_number)
                cls = cls[cls_mask].view(-1, self.class_number)

                loss_cls = self.class_weight * self.ce_loss(cls, tcls)
                # cls_softmax = F.softmax(cls, 1)
                # t_ind = torch.unsqueeze(tcls, 1).expand_as(cls_softmax)
                # class_prob = torch.gather(cls_softmax, 1, t_ind)[:, 0]
            else:
                # class_prob = torch.tensor(0.0, device=device)
                loss_cls = torch.tensor(0.0, device=device)

            self.loss_info['cls_loss'] = loss_cls.item()
            self.loss_info['obj_loss'] = loss_conf_pos.item()
            self.loss_info['no_obj_loss'] = loss_conf_neg.item()
            self.loss_info['coord_xy_loss'] = loss_coord_center.item()
            self.loss_info['coord_wh_loss'] = loss_coord_wh.item()

            all_loss = loss_coord + loss_conf + loss_cls
            return all_loss
