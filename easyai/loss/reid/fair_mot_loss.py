#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.loss.utility.loss_registry import REGISTERED_REID_LOSS
import math


# class FairMotLoss(torch.nn.Module):
#     def __init__(self, opt):
#         super().__init__()
#         self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
#         self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
#             RegLoss() if opt.reg_loss == 'sl1' else None
#         self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
#             NormRegL1Loss() if opt.norm_wh else \
#                 RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
#         self.opt = opt
#         self.emb_dim = opt.reid_dim
#         self.nID = opt.nID
#         self.classifier = nn.Linear(self.emb_dim, self.nID)
#         if opt.id_loss == 'focal':
#             torch.nn.init.normal_(self.classifier.weight, std=0.01)
#             prior_prob = 0.01
#             bias_value = -math.log((1 - prior_prob) / prior_prob)
#             torch.nn.init.constant_(self.classifier.bias, bias_value)
#         self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
#         self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
#         self.s_det = nn.Parameter(-1.85 * torch.ones(1))
#         self.s_id = nn.Parameter(-1.05 * torch.ones(1))
#
#     def forward(self, outputs, batch):
#         opt = self.opt
#         hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
#         for s in range(opt.num_stacks):
#             output = outputs[s]
#             if not opt.mse_loss:
#                 output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)
#
#             hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
#             if opt.wh_weight > 0:
#                 wh_loss += self.crit_reg(
#                     output['wh'], batch['reg_mask'],
#                     batch['ind'], batch['wh']) / opt.num_stacks
#
#             if opt.reg_offset and opt.off_weight > 0:
#                 off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
#                                           batch['ind'], batch['reg']) / opt.num_stacks
#
#             if opt.id_weight > 0:
#                 id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
#                 id_head = id_head[batch['reg_mask'] > 0].contiguous()
#                 id_head = self.emb_scale * F.normalize(id_head)
#                 id_target = batch['ids'][batch['reg_mask'] > 0]
#
#                 id_output = self.classifier(id_head).contiguous()
#                 if self.opt.id_loss == 'focal':
#                     id_target_one_hot = id_output.new_zeros((id_head.size(0), self.nID)).scatter_(1,
#                                                                                                   id_target.long().view(
#                                                                                                       -1, 1), 1)
#                     id_loss += sigmoid_focal_loss_jit(id_output, id_target_one_hot,
#                                                       alpha=0.25, gamma=2.0, reduction="sum"
#                                                       ) / id_output.size(0)
#                 else:
#                     id_loss += self.IDLoss(id_output, id_target)
#
#         det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
#         if opt.multi_loss == 'uncertainty':
#             loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
#             loss *= 0.5
#         else:
#             loss = det_loss + 0.1 * id_loss
#
#         loss_stats = {'loss': loss, 'hm_loss': hm_loss,
#                       'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
#         return loss
