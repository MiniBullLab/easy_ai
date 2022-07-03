#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_DET2D_LOSS
import math

__all__ = ['YoloV5Loss']


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


@REGISTERED_DET2D_LOSS.register_module(LossName.YoloV5Loss)
class YoloV5Loss(BaseLoss):
    def __init__(self, class_number, anchor_count, anchor_sizes,
                 box_weight=0.05, object_weight=1.0, class_weight=0.5):
        super().__init__(LossName.YoloV5Loss)
        # paramater
        anchor_sizes_str = (x for x in anchor_sizes.split('|') if x.strip())
        temp_anchor = []
        for i, data in enumerate(anchor_sizes_str, 1):
            temp_value = [float(x) for x in data.split(',') if x.strip()]
            temp_anchor.append(temp_value)
        self.stride = torch.tensor([8, 16, 32]).float()
        self.nl = len(temp_anchor) // anchor_count
        self.register_buffer('anchors', torch.tensor(temp_anchor).float().view(self.nl, -1, 2) /
                             self.stride.view(-1, 1, 1))
        self.class_number = class_number
        self.anchor_count = anchor_count
        self.no = class_number + 5  # number of outputs per anchor
        self.cls_bceloss_pos_weight = 1.0
        self.obj_bceloss_pos_weight = 1.0
        self.box_loss_gain = box_weight * (3.0 / self.nl)
        self.obj_loss_gain = object_weight  # (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.cls_loss_gain = class_weight * (self.class_number / 80.0 * 3.0 / self.nl)
        self.balance = [4.0, 1.0, 0.4]
        self.gr = 1.0
        self.fl_gamma = 0.0
        self.anchor_threshold = 4.0
        self.sort_obj_iou = False
        self.autobalance = False
        # loss defines
        self.cls_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.cls_bceloss_pos_weight))
        self.obj_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.obj_bceloss_pos_weight))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0)  # positive, negative BCE targets

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid

        if self.fl_gamma > 0:
            self.cls_bce_loss, self.obj_bce_loss = FocalLoss(self.cls_bce_loss, self.fl_gamma), \
                                                   FocalLoss(self.obj_bce_loss, self.fl_gamma)

        self.loss_info = {'cls_loss': 0.0, 'obj_loss': 0.0, 'box_loss': 0.0}

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.anchor_count, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.anchor_count, 1, 1, 2)).expand((1, self.anchor_count, ny, nx, 2)).float()
        return grid, anchor_grid

    def forward(self, outputs, batch_data=None):
        device = outputs[0].device
        output_list = []
        for i in range(len(outputs)):
            # print(outputs[i].shape)
            bs, _, ny, nx = outputs[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            output_list.append(outputs[i].view(bs, self.anchor_count, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous())
        if batch_data is None:
            z = []  # inference output
            for i in range(len(output_list)):
                bs, _, ny, nx, _ = output_list[i].shape
                if self.grid[i].shape[2:4] != output_list[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = output_list[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                z.append(y.view(bs, -1, self.no))
            return torch.cat(z, 1)
        else:
            targets = batch_data['label'].to(device)
            lcls, lbox, lobj = torch.zeros(1, device=device), \
                               torch.zeros(1, device=device), \
                               torch.zeros(1, device=device)
            tcls, tbox, indices, anchors = self.build_targets(output_list, targets)  # targets

            # Losses
            for i, pi in enumerate(output_list):  # layer index, layer predictions
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

                n = b.shape[0]  # number of targets
                if n:
                    ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                    # Regression
                    pxy = ps[:, :2].sigmoid() * 2. - 0.5
                    pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                    pbox = torch.cat((pxy, pwh), 1)  # predicted box
                    iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                    lbox += (1.0 - iou).mean()  # iou loss

                    # Objectness
                    score_iou = iou.detach().clamp(0).type(tobj.dtype)
                    if self.sort_obj_iou:
                        sort_id = torch.argsort(score_iou)
                        b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                    tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                    # Classification
                    if self.class_number > 1:  # cls loss (only if multiple classes)
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls += self.cls_bce_loss(ps[:, 5:], t)  # BCE

                    # Append targets to text file
                    # with open('targets.txt', 'a') as file:
                    #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                obji = self.obj_bce_loss(pi[..., 4], tobj)
                lobj += obji * self.balance[i]  # obj loss
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

            if self.autobalance:
                self.balance = [x / self.balance[self.ssi] for x in self.balance]
            lbox *= self.box_loss_gain
            lobj *= self.obj_loss_gain
            lcls *= self.cls_loss_gain
            bs = tobj.shape[0]  # batch size

            self.loss_info['cls_loss'] = lcls.item()
            self.loss_info['obj_loss'] = lobj.item()
            self.loss_info['box_loss'] = lbox.item()

            return (lbox + lobj + lcls) * bs

    def build_targets(self, p, targets):
        # input targets(image,class,x,y,w,h)
        na, nt = self.anchor_count, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_threshold  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
