#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np


class CalculateRectAP():

    def __init__(self):
        self.use_07_metric = False

    def calculate_ap(self, gt_boxes, detect_boxes, iou_thresh=0.5):
        if len(detect_boxes) == 0 and len(gt_boxes) == 0:
            return 1, 1, 1
        elif len(detect_boxes) == 0:
            return 0, 0, 0
        else:
            class_recs, npos = self.process_gt_boxes(gt_boxes)
            image_ids, sorted_scores, boxes = self.process_detect_result(detect_boxes)
            tp, fp, iou = self.get_tp_fp(image_ids, class_recs, boxes, iou_thresh)

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            if npos == 0:
                recall = tp
            else:
                recall = tp / float(npos)
            # avg_iou = sum(iou) / len(iou)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.get_ap(recall, precision)
            return recall, precision, ap

    def process_gt_boxes(self, gt_boxes):
        class_recs = {}
        npos = 0
        for filename_post, boxes in gt_boxes.items():
            bbox = np.array([x.getVector() for x in boxes])
            difficult = np.array([x.difficult for x in boxes]).astype(np.bool)
            det = [False] * len(boxes)
            npos = npos + sum(~difficult)
            class_recs[filename_post] = {'bbox': bbox,
                                         'difficult': difficult,
                                         'det2d': det}
        return class_recs, npos

    def process_detect_result(self, detect_boxes):
        image_ids = []
        confidence = []
        boxes = []
        for image_id, temp_object in detect_boxes:
            image_ids.append(image_id)
            confidence.append(temp_object.objectConfidence)
            boxes.append([
                float(temp_object.min_corner.x),
                float(temp_object.min_corner.y),
                float(temp_object.max_corner.x),
                float(temp_object.max_corner.y)
            ])
        confidence = np.array(confidence)
        boxes = np.array(boxes)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        boxes = boxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        return image_ids, sorted_scores, boxes

    def calculate_iou(self, BBGT, bb):
        ovmax = -np.inf
        jmax = None
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        return ovmax, jmax

    def get_tp_fp(self, image_ids, class_recs, BB, iou_thresh):
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        iou = []
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            BBGT = R['bbox'].astype(float)
            ovmax, jmax = self.calculate_iou(BBGT, bb)
            if ovmax > iou_thresh:
                if not R['difficult'][jmax]:
                    if not R['det2d'][jmax]:
                        tp[d] = 1.
                        R['det2d'][jmax] = 1
                        iou.append(ovmax)
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.
        return tp, fp, iou

    def get_ap(self, recall, precision):
        """
        ap = voc_ap(rec, prec, [use_07_metric])
                Compute VOC AP given precision and recall.
                If use_07_metric is true, uses the
                VOC 07 11 point method (default:False).
        """
        if self.use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], recall, [1.]))
            mpre = np.concatenate(([0.], precision, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
