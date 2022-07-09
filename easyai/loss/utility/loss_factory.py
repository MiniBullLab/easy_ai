#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.loss_name import LossName
from easyai.loss.utility.loss_registry import REGISTERED_COMMON_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_CLS_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_DET2D_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_SEG_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_GAN_D_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_GAN_G_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_KEYPOINT2D_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_RNN_LOSS
from easyai.loss.utility.loss_registry import REGISTERED_REID_LOSS
from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger


class LossFactory():

    def __init__(self):
        pass

    def get_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        loss_args = loss_config.copy()
        EasyLogger.debug(loss_args)
        if REGISTERED_COMMON_LOSS.has_class(input_name):
            result = self.get_common_loss(loss_args)
        elif REGISTERED_CLS_LOSS.has_class(input_name):
            result = self.get_cls_loss(loss_args)
        elif REGISTERED_DET2D_LOSS.has_class(input_name):
            result = self.get_det2d_loss(loss_args)
        elif REGISTERED_SEG_LOSS.has_class(input_name):
            result = self.get_seg_loss(loss_args)
        elif REGISTERED_KEYPOINT2D_LOSS.has_class(input_name):
            result = self.get_keypoint2d_loss(loss_args)
        elif REGISTERED_RNN_LOSS.has_class(input_name):
            result = self.get_rnn_loss(loss_args)
        elif REGISTERED_REID_LOSS.has_class(input_name):
            result = self.get_reid_loss(loss_args)
        else:
            result = self.get_gan_loss(loss_args)
        if result is None:
            EasyLogger.error("loss:%s error" % input_name)
        return result

    def has_loss(self, key):

        for loss_name in REGISTERED_COMMON_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_CLS_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_DET2D_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_SEG_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_KEYPOINT2D_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_GAN_D_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_GAN_G_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_RNN_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_REID_LOSS.get_keys():
            if loss_name in key:
                return True

        return False

    def get_loss_input(self, key, model_output, multi_output):
        loss_input = []
        if LossName.MultiBoxLoss in key:
            loss_input.extend(multi_output)
        elif LossName.RPNLoss in key:
            loss_input.append(multi_output)
        elif LossName.FastRCNNLoss in key:
            loss_input.append(multi_output)
        elif LossName.Keypoint2dRCNNLoss:
            loss_input.append(model_output)
        else:
            loss_input.append(model_output)
        return loss_input

    def get_common_loss(self, loss_config):
        loss = build_from_cfg(loss_config, REGISTERED_COMMON_LOSS)
        return loss

    def get_cls_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.CrossEntropy2dLoss:
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            weight_str = loss_config.get("weight", None)
            if weight_str is not None:
                weights = [float(x) for x in weight_str.split(',') if x]
                loss_config['weight'] = weights
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.BinaryCrossEntropy2dLoss:
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            weight_str = loss_config.get("weight", None)
            if weight_str is not None:
                weights = [float(x) for x in weight_str.split(',') if x]
                loss_config['weight'] = weights
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.OhemCrossEntropy2dLoss:
            loss_config['threshold'] = float(loss_config.get("ignore_index", 0.7))
            loss_config['min_keep'] = int(loss_config.get("min_keep", int(32 // 1 * 640 * 352 // 16)))
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.OhemBinaryCrossEntropy2dLoss:
            loss_config['threshold'] = float(loss_config.get("ignore_index", 0.7))
            loss_config['min_keep'] = int(loss_config.get("min_keep", int(32 // 1 * 640 * 352 // 16)))
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.CenterCrossEntropy2dLoss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['feature_dim'] = int(loss_config.get('feature_dim', 2))
            loss_config['alpha'] = int(loss_config.get('alpha', 1))
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        loss = build_from_cfg(loss_config, REGISTERED_CLS_LOSS)
        return loss

    def get_det2d_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.Region2dLoss:
            anchor_sizes_str = (x for x in loss_config['anchor_sizes'].split('|') if x.strip())
            anchor_sizes = []
            for data in anchor_sizes_str:
                temp_value = [float(x) for x in data.split(',') if x.strip()]
                anchor_sizes.append(temp_value)
            loss_config['anchor_sizes'] = anchor_sizes
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['reduction'] = int(loss_config['reduction'])
            loss_config['coord_weight'] = float(loss_config['coord_weight'])
            loss_config['noobject_weight'] = float(loss_config['noobject_weight'])
            loss_config['object_weight'] = float(loss_config['object_weight'])
            loss_config['class_weight'] = float(loss_config['class_weight'])
            loss_config['iou_threshold'] = float(loss_config['iou_threshold'])
        elif input_name == LossName.YoloV3Loss:
            anchor_mask = [int(x) for x in loss_config['anchor_mask'].split(',')]
            loss_config['anchor_mask'] = anchor_mask
            anchor_sizes_str = (x for x in loss_config['anchor_sizes'].split('|') if x.strip())
            anchor_sizes = []
            for data in anchor_sizes_str:
                temp_value = [float(x) for x in data.split(',') if x.strip()]
                anchor_sizes.append(temp_value)
            loss_config['anchor_sizes'] = anchor_sizes
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['reduction'] = int(loss_config['reduction'])
            loss_config['coord_weight'] = float(loss_config['coord_weight'])
            loss_config['noobject_weight'] = float(loss_config['noobject_weight'])
            loss_config['object_weight'] = float(loss_config['object_weight'])
            loss_config['class_weight'] = float(loss_config['class_weight'])
            loss_config['iou_threshold'] = float(loss_config['iou_threshold'])
        elif input_name == LossName.MultiBoxLoss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['iou_threshold'] = float(loss_config['iou_threshold'])
            loss_config['input_size'] = tuple(int(x) for x in
                                              loss_config['input_size'].split(',') if x.strip())
            loss_config['anchor_counts'] = tuple(int(x) for x in
                                                 loss_config['anchor_counts'].split(',') if x.strip())
            aspect_ratio_str = tuple(x for x in loss_config['aspect_ratios'].split('|') if x.strip())
            aspect_ratio_list = []
            for data in aspect_ratio_str:
                temp_value = [int(x) for x in data.split(',') if x.strip()]
                aspect_ratio_list.append(temp_value)
            loss_config['aspect_ratios'] = aspect_ratio_list
            loss_config['min_sizes'] = tuple(int(x) for x in
                                             loss_config['min_sizes'].split(',') if x.strip())
            max_sizes_str = loss_config.get("max_sizes", None)
            if max_sizes_str is not None:
                loss_config['max_sizes'] = tuple(int(x) for x in
                                                 loss_config['max_sizes'].split(',') if x.strip())
        elif input_name == LossName.RPNLoss:
            loss_config['input_size'] = tuple(int(x) for x in
                                              loss_config['input_size'].split(',') if x.strip())
            loss_config['anchor_sizes'] = tuple(int(x) for x in
                                                loss_config['anchor_sizes'].split(',') if x.strip())
            loss_config['anchor_sizes'] = tuple(float(x) for x in loss_config['aspect_ratios'].split(',')
                                                if x.strip())
            loss_config['anchor_strides'] = tuple(int(x) for x in loss_config['anchor_strides'].split(',')
                                                  if x.strip())
            loss_config['fg_iou_threshold'] = float(loss_config['fg_iou_threshold'])
            loss_config['bg_iou_threshold'] = float(loss_config['bg_iou_threshold'])
            loss_config['per_image_sample'] = int(loss_config['per_image_sample'])
            loss_config['positive_fraction'] = float(loss_config['positive_fraction'])
        elif input_name == LossName.FastRCNNLoss:
            loss_config['fg_iou_threshold'] = float(loss_config['fg_iou_threshold'])
            loss_config['bg_iou_threshold'] = float(loss_config['bg_iou_threshold'])
            loss_config['per_image_sample'] = int(loss_config['per_image_sample'])
            loss_config['positive_fraction'] = float(loss_config['positive_fraction'])
        elif input_name == LossName.YoloV5Loss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['anchor_count'] = int(loss_config['anchor_count'])
            loss_config['anchor_sizes'] = loss_config['anchor_sizes']
            loss_config['box_weight'] = float(loss_config['box_weight'])
            loss_config['object_weight'] = float(loss_config['object_weight'])
            loss_config['class_weight'] = float(loss_config['class_weight'])
        loss = build_from_cfg(loss_config, REGISTERED_DET2D_LOSS)
        return loss

    def get_seg_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.EncNetLoss:
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.MixCrossEntropy2dLoss:
            loss_config['aux_weight'] = float(loss_config.get("aux_weight", 0.2))
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            weight_str = loss_config.get("weight", None)
            if weight_str is not None:
                weights = [float(x) for x in weight_str.split(',') if x]
                loss_config['weight'] = weights
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.MixBinaryCrossEntropy2dLoss:
            loss_config['aux_weight'] = float(loss_config.get("aux_weight", 0.2))
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            weight_str = loss_config.get("weight", None)
            if weight_str is not None:
                weights = [float(x) for x in weight_str.split(',') if x]
                loss_config['weight'] = weights
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.DBLoss:
            loss_config['alpha'] = loss_config.get('alpha', 1.0)
            loss_config['beta'] = loss_config.get('beta', 10.0)
            loss_config['ohem_ratio'] = loss_config.get('ohem_ratio', 3)
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
        loss = build_from_cfg(loss_config, REGISTERED_SEG_LOSS)
        return loss

    def get_keypoint2d_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.Keypoint2dRegionLoss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['point_count'] = int(loss_config['point_count'])
        elif input_name == LossName.JointsMSELoss:
            loss_config['reduction'] = int(loss_config['reduction'])
            loss_config['input_size'] = tuple(int(x) for x in
                                              loss_config['input_size'].split(',') if x.strip())
            loss_config['points_count'] = int(loss_config['points_count'])
        elif input_name == LossName.FaceLandmarkLoss:
            loss_config['input_size'] = tuple(int(x) for x in
                                              loss_config['input_size'].split(',') if x.strip())
            loss_config['points_count'] = int(loss_config['points_count'])
            loss_config['wing_w'] = float(loss_config['wing_w'])
            loss_config['wing_e'] = float(loss_config['wing_e'])
            loss_config['gaussian_scale'] = float(loss_config['gaussian_scale'])
            loss_config['ignore_value'] = int(loss_config.get("ignore_value", -1000))
        loss = build_from_cfg(loss_config, REGISTERED_KEYPOINT2D_LOSS)
        return loss

    def get_gan_loss(self, loss_config):
        loss = None
        input_name = loss_config['type'].strip()
        if REGISTERED_GAN_D_LOSS.has_class(input_name):
            loss = build_from_cfg(loss_config, REGISTERED_GAN_D_LOSS)
        elif REGISTERED_GAN_G_LOSS.has_class(input_name):
            loss = build_from_cfg(loss_config, REGISTERED_GAN_G_LOSS)
        return loss

    def get_rnn_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.CTCLoss:
            loss_config['blank_index'] = int(loss_config['blank_index'])
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['use_focal'] = bool(loss_config.get("use_focal", False))
        elif input_name == LossName.ACELabelSmoothingLoss:
            loss_config['alpha'] = float(loss_config.get('alpha', 0.1))
        loss = build_from_cfg(loss_config, REGISTERED_RNN_LOSS)
        return loss

    def get_reid_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.FairMotLoss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['reid'] = int(loss_config['reid'])
            loss_config['max_id'] = int(loss_config['max_id'])
        loss = build_from_cfg(loss_config, REGISTERED_REID_LOSS)
        return loss



