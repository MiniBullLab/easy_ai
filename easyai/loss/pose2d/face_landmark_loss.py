#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.loss_name import LossName
from easyai.loss.utility.base_loss import *
from easyai.model.utility.model_factory import ModelFactory
from easyai.loss.common.wing_loss import WingLoss
from easyai.loss.common.common_loss import GaussianNLLoss
from easyai.loss.pose2d.mouth_eye_dis_loss import MouthEyeFrontDisLoss
from easyai.loss.pose2d.mouth_eye_dis_loss import MouthEyeProfierDisLoss
from easyai.loss.utility.registry import REGISTERED_POSE2D_LOSS


@REGISTERED_POSE2D_LOSS.register_module(LossName.FaceLandmarkLoss)
class FaceLandmarkLoss(BaseLoss):

    def __init__(self, input_size, points_count,
                 wing_w=15, wing_e=3, gaussian_scale=4,
                 ignore_value=-1):
        super().__init__(LossName.FaceLandmarkLoss)
        self.input_size = input_size
        self.points_count = points_count
        self.coords_loss1 = WingLoss(wing_w, wing_e, ignore_value=ignore_value)
        self.mouth_eye_loss1 = MouthEyeFrontDisLoss(ignore_value=ignore_value)

        self.coords_loss2 = WingLoss(wing_w, wing_e, ignore_value=ignore_value)
        self.mouth_eye_loss2 = MouthEyeProfierDisLoss(ignore_value=ignore_value)

        self.coords_loss3 = WingLoss(wing_w, wing_e, ignore_value=ignore_value)
        self.mouth_eye_loss3 = MouthEyeProfierDisLoss(ignore_value=ignore_value)

        self.direction_loss = nn.CrossEntropyLoss(reduction='mean')
        self.box_loss = WingLoss(wing_w, wing_e, ignore_value=-1000)
        self.conf_loss = nn.SmoothL1Loss(reduction='mean')
        self.gaussian_loss = GaussianNLLoss(gaussian_scale, reduction='mean',
                                            ignore_value=ignore_value)

        model_factory = ModelFactory()
        model_config = {"type": ModelName.HourglassPose,
                        "data_channel": 1,
                        "points_count": self.points_count}
        self.hm_model = model_factory.get_model(model_config)

        self.convert_left = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
                             [6, 6], [7, 7], [8, 8],
                             [9, 17], [10, 18], [11, 19], [12, 20], [13, 21],
                             [14, 36], [15, 37], [16, 38], [17, 39],
                             [18, 40], [19, 41], [20, 27], [21, 28],
                             [22, 29], [23, 30], [24, 31], [25, 32], [26, 33],
                             [27, 51], [28, 50], [29, 49], [30, 48],
                             [31, 59], [32, 58], [33, 57], [34, 66], [35, 67],
                             [36, 60], [37, 61], [38, 62]]

        self.convert_right = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12],
                              [5, 11], [6, 10], [7, 9], [8, 8],
                              [9, 26], [10, 25], [11, 24], [12, 23], [13, 22],
                              [14, 45], [15, 44], [16, 43], [17, 42],
                              [18, 47], [19, 46], [20, 27], [21, 28], [22, 29],
                              [23, 30], [24, 35], [25, 34], [26, 33],
                              [27, 51], [28, 52], [29, 53], [30, 54], [31, 55],
                              [32, 56], [33, 57], [34, 66], [35, 65],
                              [36, 64], [37, 63], [38, 62]]
        assert len(self.convert_left) == len(self.convert_right)
        self.left_count = len(self.convert_left)

        self.loss_info = {'coord_loss': 0.0, 'coord_left_loss': 0.0,
                          'coord_right_loss': 0.0, 'direction_loss': 0.0,
                          'box_loss': 0.0, 'conf_loss': 0.0, 'gaussian_loss': 0.0}

    def build_left_and_right_coords(self, gt_coords, device):
        left_coords = []
        right_coords = []
        for coordinate in gt_coords:
            left_out = torch.zeros(self.left_count*2, dtype=torch.float, device=device)
            right_out = torch.zeros(self.left_count * 2, dtype=torch.float, device=device)
            for index in self.convert_left:
                left_out[2 * index[0]] = coordinate[2 * index[1]]
                left_out[2 * index[0] + 1] = coordinate[2 * index[1] + 1]
            for index in self.convert_right:
                right_out[2 * index[0]] = coordinate[2 * index[1]]
                right_out[2 * index[0] + 1] = coordinate[2 * index[1] + 1]
            left_coords.append(left_out)
            right_coords.append(right_out)
        return torch.stack(left_coords, dim=0).to(device), \
               torch.stack(right_coords, dim=0).to(device)

    def get_preds(self, scores):
        ''' get predictions from score maps in torch Tensor
            return type: torch.LongTensor
        '''
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        # batch, chn, height, width ===> batch, chn, height*width
        # chn = 68
        # height*width = score_map
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        # batchsize * numPoints * 2
        # 0 is x coord
        # 1 is y coord
        # shape = batchsize, numPoints, 2
        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds, maxval.view(scores.size(0), scores.size(1))

    def conf_norm(self, conf, conf_sub_value):
        conf = conf - conf_sub_value
        temp_0 = torch.exp(conf * 5)
        temp_1 = torch.exp(5 - conf * 5)
        conf_norm_value = temp_0 / (temp_0 + temp_1)
        # print('conf max: {} | min: {} | mean: {}'.format(conf_norm_value.max(),
        #                                                  conf_norm_value.min(),
        #                                                  conf_norm_value.mean()))
        return conf_norm_value

    def get_confidence_gt(self, input_image, device):
        with torch.no_grad():
            self.hm_model.to(device)
            self.hm_model.eval()
            output_list = self.hm_model(input_image)
            score_map = output_list[-1].detach()
            _, gt_conf = self.get_preds(score_map)
            conf_sub_value = torch.ones_like(gt_conf, dtype=torch.float).to(gt_conf.device)
            gt_conf = self.conf_norm(gt_conf, conf_sub_value)
        return gt_conf.to(device)

    def forward(self, outputs, targets=None):
        """
        Arguments:
            outputs (Tensor))
            targets (Tensor, Tensor)

        Returns:
            loss (Tensor)
        """
        ldmk = outputs[1]
        conf = outputs[4]
        gauss = outputs[5]
        device = ldmk.device
        batch_size = ldmk.size(0)
        if targets is None:
            final_gauss = torch.zeros_like(conf, dtype=torch.float).to(device)
            for i in range(self.points_count):
                final_gauss[:, i] = 1.0 - 0.5 * (gauss[:, 2 * i] + gauss[:, 2 * i + 1])
            landmark_conf = conf * final_gauss
            return ldmk, landmark_conf
        else:
            input_image = outputs[0]
            left_ldmk = outputs[2]
            right_ldmk = outputs[3]
            pre_direction_cls = outputs[6]
            pre_box = outputs[7]
            gt_coords = targets[0]
            gt_face_box = targets[1][:, :4]
            gt_direction_cls = targets[1][:, 4]
            gt_coords = gt_coords.reshape((batch_size, -1)).to(device)
            left_coords, right_coords = self.build_left_and_right_coords(gt_coords.detach(), device)
            gt_conf = self.get_confidence_gt(input_image, device)

            loss1_1 = self.coords_loss1(ldmk, gt_coords)
            loss1_2 = self.mouth_eye_loss1(ldmk, gt_coords)
            loss1 = loss1_1 + loss1_2

            loss2_1 = self.coords_loss2(left_ldmk, left_coords)
            loss2_2 = self.mouth_eye_loss2(left_ldmk, left_coords)
            loss2 = loss2_1 + loss2_2

            loss3_1 = self.coords_loss3(right_ldmk, right_coords)
            loss3_2 = self.mouth_eye_loss3(right_ldmk, right_coords)
            loss3 = loss3_1 + loss3_2

            loss4 = self.direction_loss(pre_direction_cls, gt_direction_cls)
            loss5 = self.box_loss(pre_box, gt_face_box)
            loss6 = self.conf_loss(conf, gt_conf)
            loss7 = self.gaussian_loss([ldmk, gauss], gt_coords)

            self.loss_info['coord_loss'] = loss1.item()
            self.loss_info['coord_left_loss'] = loss2.item()
            self.loss_info['coord_right_loss'] = loss3.item()
            self.loss_info['direction_loss'] = loss4.item()
            self.loss_info['box_loss'] = loss5.item()
            self.loss_info['conf_loss'] = 0
            self.loss_info['gaussian_loss'] = loss7.item()

            all_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 * 100. + loss7
            return all_loss
