#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from easyai.helper.data_structure import ReIDObject2d
from easy_tracking.fairmot.image import transform_preds


class FairMOTPostProcess():

    def __init__(self, image_size, class_number, threshold=0.4):
        super().__init__()
        self.image_size = image_size
        self.class_number = class_number
        self.threshold = threshold
        self.K = 500
        self.ltrb = True
        self.down_ratio = 4

    def set_threshold(self, value):
        self.threshold = value

    def __call__(self, output_list, src_size):
        result = []
        assert len(output_list) == 4
        c = np.array([src_size[0] / 2., src_size[1] / 2.], dtype=np.float32)
        s = max(float(self.image_size[0]) / float(self.image_size[0]) * src_size[1], src_size[0]) * 1.0
        meta = {'c': c, 's': s,
                'out_height': self.image_size[1] // self.down_ratio,
                'out_width': self.image_size[0] // self.down_ratio}
        hm = output_list[0].sigmoid_()
        id_feature = output_list[1]
        id_feature = F.normalize(id_feature, dim=1)
        reg = output_list[2]
        wh = output_list[3]
        dets, inds = self.mot_decode(hm, wh, reg=reg, ltrb=self.ltrb, K=self.K)
        id_feature = self._tranpose_and_gather_feat(id_feature, inds)
        id_feature = id_feature.squeeze(0)
        id_feature = id_feature.cpu().numpy()
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = self.ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.class_number)
        for j in range(1, self.class_number + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets = dets[0]
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.threshold
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        for index, det in enumerate(dets):
            temp = ReIDObject2d()
            temp.min_corner.x = det[0]
            temp.min_corner.y = det[1]
            temp.max_corner.x = det[2]
            temp.max_corner.y = det[3]
            temp.classConfidence = det[4]
            temp.reid = id_feature[index]
            result.append(temp)
        return result

    def mot_decode(self, heat, wh, reg=None, ltrb=False, K=100):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        if reg is not None:
            reg = self._tranpose_and_gather_feat(reg, inds)
            # print(reg.shape)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self._tranpose_and_gather_feat(wh, inds)
        if ltrb:
            wh = wh.view(batch, K, 4)
        else:
            wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        if ltrb:
            bboxes = torch.cat([xs - wh[..., 0:1],
                                ys - wh[..., 1:2],
                                xs + wh[..., 2:3],
                                ys + wh[..., 3:4]], dim=2)
        else:
            bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                                ys - wh[..., 1:2] / 2,
                                xs + wh[..., 0:1] / 2,
                                ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections, inds

    def ctdet_post_process(self, dets, c, s, h, w, num_classes):
        # dets: batch x max_dets x dim
        # return 1-based class det dict
        ret = []
        for i in range(dets.shape[0]):
            top_preds = {}
            dets[i, :, :2] = transform_preds(
                dets[i, :, 0:2], c[i], s[i], (w, h))
            dets[i, :, 2:4] = transform_preds(
                dets[i, :, 2:4], c[i], s[i], (w, h))
            classes = dets[i, :, -1]
            for j in range(num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
            ret.append(top_preds)
        return ret

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.class_number + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.class_number + 1)])
        if len(scores) > self.K:
            kth = len(scores) - self.K
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.class_number + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = torch.true_divide(topk_inds, width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = torch.true_divide(topk_ind, K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
