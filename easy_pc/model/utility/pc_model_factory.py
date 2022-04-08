#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.registry import build_from_cfg
from easyai.model.utility.mode_weight_init import ModelWeightInit
from easyai.utility.logger import EasyLogger

from easy_pc.model.utility.pc_model_registry import REGISTERED_PC_CLS_MODEL
from easy_pc.model.utility.pc_model_registry import REGISTERED_PC_SEG_MODEL


class PCModelFactory():

    def __init__(self):
        self.model_weight_init = ModelWeightInit()

    def get_model(self, model_config):
        EasyLogger.debug(model_config)
        input_name = model_config['type'].strip()
        model_args = model_config.copy()
        result = self.get_model_from_name(model_args)
        if result is None:
            EasyLogger.error("%s model error!" % input_name)
        self.model_weight_init.init_weight(result)
        return result

    def get_model_from_name(self, model_config):
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        model_name = model_config['type'].strip()
        model_config['type'] = model_name
        EasyLogger.debug(model_config)
        if REGISTERED_PC_CLS_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_PC_CLS_MODEL)
        elif REGISTERED_PC_SEG_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_PC_SEG_MODEL)
        else:
            model = None
        return model
