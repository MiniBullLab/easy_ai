#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
from easyai.utility.registry import build_from_cfg
from easy_pc.model_block.utility.pc_block_registry import REGISTERED_PC_CLS_BACKBONE
from easy_pc.model_block.utility.pc_block_registry import REGISTERED_PC_DET3D_BACKBONE

__all__ = ["PCBackboneFactory"]


class PCBackboneFactory():

    def __init__(self):
        pass

    def get_backbone_model(self, model_config):
        result = None
        EasyLogger.debug(model_config)
        try:
            input_name = model_config['type'].strip()
            model_args = model_config.copy()
            result = self.get_pc_backbone_from_name(model_args)
            if result is None:
                EasyLogger.error("backbone:%s error" % input_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result

    def get_pc_backbone_from_name(self, model_config):
        result = None
        model_args = model_config.copy()
        input_name = model_args['type'].strip()
        if REGISTERED_PC_CLS_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_args, REGISTERED_PC_CLS_BACKBONE)
        elif REGISTERED_PC_DET3D_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_args, REGISTERED_PC_DET3D_BACKBONE)
        return result
