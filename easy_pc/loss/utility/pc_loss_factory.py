#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.registry import build_from_cfg
from easyai.utility.logger import EasyLogger

from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_CLS_LOSS
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_DET3D_LOSS
from easy_pc.loss.utility.pc_loss_registry import REGISTERED_PC_SEG_LOSS


class PCLossFactory():

    def __init__(self):
        pass

    def get_loss(self, loss_config):
        result = None
        input_name = loss_config['type'].strip()
        loss_args = loss_config.copy()
        EasyLogger.debug(loss_args)
        if REGISTERED_PC_CLS_LOSS.has_class(input_name):
            result = build_from_cfg(loss_config, REGISTERED_PC_CLS_LOSS)
        elif REGISTERED_PC_DET3D_LOSS.has_class(input_name):
            result = build_from_cfg(loss_config, REGISTERED_PC_DET3D_LOSS)
        elif REGISTERED_PC_SEG_LOSS.has_class(input_name):
            result = build_from_cfg(loss_config, REGISTERED_PC_SEG_LOSS)
        if result is None:
            EasyLogger.error("loss:%s error" % input_name)
        return result

    def has_loss(self, key):

        for loss_name in REGISTERED_PC_CLS_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_PC_DET3D_LOSS.get_keys():
            if loss_name in key:
                return True

        for loss_name in REGISTERED_PC_SEG_LOSS.get_keys():
            if loss_name in key:
                return True

        return False


