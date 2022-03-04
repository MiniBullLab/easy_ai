#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_REID_LOSS
import math
import copy
import numpy as np


class RegL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, mask, ind, target):
        pred = self.tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def tranpose_and_gather_feat(self, feat, ind):
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


@REGISTERED_REID_LOSS.register_module(LossName.FairMotLoss)
class FairMotLoss(torch.nn.Module):
    def __init__(self, class_number, reid, max_id):
        super().__init__()
        self.class_number = class_number
        self.emb_dim = reid
        self.max_id = max_id
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.classifier = nn.Linear(self.emb_dim, self.max_id)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.max_id - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        self.max_objs = 500
        self.wh_weight = 0.5
        self.off_weight = 1
        self.id_weight = 1

        self.loss_info = {'hm_loss': 0.0, 'wh_loss': 0.0,
                          'off_loss': 0.0, 'id_loss': 0.0}

    def build_targets(self, output_w, output_h, device, batch_data):
        hm_list = []
        reg_mask_list = []
        ind_list = []
        wh_list = []
        reg_list = []
        ids_list = []
        for labels in batch_data["label"]:
            num_objs = labels.shape[0]
            hm = np.zeros((self.class_number, output_h, output_w), dtype=np.float32)
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
            reg = np.zeros((self.max_objs, 2), dtype=np.float32)
            ind = np.zeros((self.max_objs,), dtype=np.int64)
            reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)
            ids = np.zeros((self.max_objs,), dtype=np.int64)

            for k in range(min(num_objs, self.max_objs)):
                label = labels[k]
                bbox = label[2:]
                cls_id = int(label[0])
                bbox[[0, 2]] = bbox[[0, 2]] * output_w
                bbox[[1, 3]] = bbox[[1, 3]] * output_h
                bbox_amodal = copy.deepcopy(bbox)
                bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
                bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
                bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
                bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
                bbox[0] = np.clip(bbox[0], 0, output_w - 1)
                bbox[1] = np.clip(bbox[1], 0, output_h - 1)
                w = bbox[2]
                h = bbox[3]

                if h > 0 and w > 0:
                    radius = self.gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    self.draw_umich_gaussian(hm[cls_id], ct_int, radius)
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                    ind[k] = ct_int[1] * output_w + ct_int[0]
                    reg[k] = ct - ct_int
                    reg_mask[k] = 1
                    ids[k] = label[1]

            hm_list.append(torch.from_numpy(hm))
            reg_mask_list.append(torch.from_numpy(reg_mask))
            ind_list.append(torch.from_numpy(ind))
            wh_list.append(torch.from_numpy(wh))
            reg_list.append(torch.from_numpy(reg))
            ids_list.append(torch.from_numpy(ids))

        ret = {'hm': torch.stack(hm_list, dim=0).to(device),
               'reg_mask': torch.stack(reg_mask_list, dim=0).to(device),
               'ind': torch.stack(ind_list, dim=0).to(device),
               'wh': torch.stack(wh_list, dim=0).to(device),
               'reg': torch.stack(reg_list, dim=0).to(device),
               'ids': torch.stack(ids_list, dim=0).to(device)}
        return ret

    def forward(self, outputs, batch_data):
        if batch_data is None:
            return outputs
        else:
            batch_size, _, height, width = outputs[0].size()
            device = outputs[0].device
            targets = self.build_targets(width, height, device, batch_data)
            hm_output = torch.clamp(outputs[0].sigmoid_(), min=1e-4, max=1 - 1e-4)
            hm_loss = self._neg_loss(hm_output, targets['hm'])
            wh_loss = self.crit_reg(outputs[3], targets['reg_mask'],
                                    targets['ind'], targets['wh'])
            off_loss = self.crit_reg(outputs[2], targets['reg_mask'],
                                     targets['ind'], targets['reg'])
            id_head = self.crit_reg.tranpose_and_gather_feat(outputs[1], targets['ind'])
            id_head = id_head[targets['reg_mask'] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = targets['ids'][targets['reg_mask'] > 0]

            id_output = self.classifier(id_head).contiguous()
            id_loss = self.IDLoss(id_output, id_target)

            det_loss = self.hm_weight * hm_loss + \
                       self.wh_weight * wh_loss + \
                       self.off_weight * off_loss
            loss = det_loss + 0.1 * id_loss

            self.loss_info['hm_loss'] = hm_loss.item()
            self.loss_info['wh_loss'] = wh_loss.item()
            self.loss_info['off_loss'] = off_loss.item()
            self.loss_info['id_loss'] = id_loss.item()

            return loss

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def draw_umich_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)
