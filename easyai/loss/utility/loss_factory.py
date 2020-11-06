#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.loss_name import LossName
from easyai.loss.utility.registry import REGISTERED_COMMON_LOSS
from easyai.loss.utility.registry import REGISTERED_CLS_LOSS
from easyai.loss.utility.registry import REGISTERED_DET2D_LOSS
from easyai.loss.utility.registry import REGISTERED_SEG_LOSS
from easyai.loss.utility.registry import REGISTERED_GAN_D_LOSS
from easyai.loss.utility.registry import REGISTERED_GAN_G_LOSS
from easyai.utility.registry import build_from_cfg


class LossFactory():

    def __init__(self):
        pass

    def get_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        loss_args = loss_config.copy()
        if REGISTERED_COMMON_LOSS.has_class(input_name):
            result = self.get_common_loss(loss_args)
        elif REGISTERED_CLS_LOSS.has_class(input_name):
            result = self.get_cls_loss(loss_args)
        elif REGISTERED_DET2D_LOSS.has_class(input_name):
            result = self.get_det2d_loss(loss_args)
        elif REGISTERED_SEG_LOSS.has_class(loss_args):
            result = self.get_seg_loss(loss_args)
        else:
            result = self.get_gan_loss(loss_args)
        if result is None:
            print("loss:%s error" % input_name)
        return result

    def has_loss(self, key):

        for loss_name in REGISTERED_COMMON_LOSS.module_dict():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_CLS_LOSS.module_dict():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_DET2D_LOSS.module_dict():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_SEG_LOSS.module_dict():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_GAN_D_LOSS.module_dict():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_GAN_G_LOSS.module_dict():
            if loss_name in key:
                return True

        return False

    def get_common_loss(self, loss_config):
        loss = build_from_cfg(loss_config, REGISTERED_SEG_LOSS)
        return loss

    def get_cls_loss(self, loss_config):
        input_name = loss_config['type'].strip()
        if input_name == LossName.CrossEntropy2d:
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            loss_config['weight'] = loss_config.get("weight", None)
            loss_config['reduce'] = loss_config.get("reduce", None)
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
            loss_config['ignore_index'] = int(loss_config.get("ignore_index", 250))
        elif input_name == LossName.BinaryCrossEntropy2d:
            loss_config['weight_type'] = int(loss_config.get("weight_type", 0))
            loss_config['weight'] = loss_config.get("weight", None)
            loss_config['reduce'] = loss_config.get("reduce", None)
            loss_config['reduction'] = loss_config.get("reduction", 'mean')
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
            loss_config['input_size'] = (int(x) for x in
                                         loss_config['input_size'].split(',') if x.strip())
            loss_config['anchor_counts'] = (int(x) for x in
                                            loss_config['anchor_counts'].split(',') if x.strip())
            loss_config['anchor_sizes'] = (int(x) for x in
                                           loss_config['anchor_sizes'].split(',') if x.strip())
            aspect_ratio_str = (x for x in loss_config['aspect_ratio_list'].split('|') if x.strip())
            aspect_ratio_list = []
            for data in aspect_ratio_str:
                temp_value = [int(x) for x in data.split(',') if x.strip()]
                aspect_ratio_list.append(temp_value)
            loss_config['aspect_ratio_list'] = aspect_ratio_list
        elif input_name == LossName.KeyPoints2dRegionLoss:
            loss_config['class_number'] = int(loss_config['class_number'])
            loss_config['point_count'] = int(loss_config['point_count'])
        loss = build_from_cfg(loss_config, REGISTERED_DET2D_LOSS)
        return loss

    def get_seg_loss(self, loss_config):
        loss = build_from_cfg(loss_config, REGISTERED_SEG_LOSS)
        return loss

    def get_gan_loss(self, loss_config):
        loss = None
        input_name = loss_config['type'].strip()
        if REGISTERED_GAN_D_LOSS.has_class(input_name):
            loss = build_from_cfg(loss_config, REGISTERED_SEG_LOSS)
        elif REGISTERED_GAN_G_LOSS.has_class(input_name):
            loss = build_from_cfg(loss_config, REGISTERED_SEG_LOSS)
        return loss

