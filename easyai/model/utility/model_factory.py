#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import os
from easyai.model.utility.model_registry import REGISTERED_CLS_MODEL
from easyai.model.utility.model_registry import REGISTERED_DET2D_MODEL
from easyai.model.utility.model_registry import REGISTERED_SEG_MODEL
from easyai.model.utility.model_registry import REGISTERED_KEYPOINT2D_MODEL
from easyai.model.utility.model_registry import REGISTERED_SR_MODEL
from easyai.model.utility.model_registry import REGISTERED_GAN_MODEL
from easyai.model.utility.model_registry import REGISTERED_RNN_MODEL
from easyai.model.utility.model_registry import REGISTERED_REID_MODEL
from easyai.model.utility.model_registry import REGISTERED_MULTI_MODEL
from easyai.model.utility.model_registry import REGISTERED_PC_CLS_MODEL
from easyai.model.utility.model_registry import REGISTERED_PC_SEG_MODEL
from easyai.utility.registry import build_from_cfg

from easyai.model_block.utility.model_parse import ModelParse
from easyai.model.common.my_model import MyModel
from easyai.model.utility.mode_weight_init import ModelWeightInit

from easyai.utility.logger import EasyLogger


class ModelFactory():

    def __init__(self):
        self.modelParse = ModelParse()
        self.model_weight_init = ModelWeightInit()

    def get_model(self, model_config):
        EasyLogger.debug(model_config)
        input_name = model_config['type'].strip()
        model_args = model_config.copy()
        if input_name.endswith("cfg"):
            model_args.pop("type")
            result = self.get_model_from_cfg(input_name, model_args)
        else:
            result = self.get_model_from_name(model_args)
            if result is None:
                EasyLogger.error("%s model error!" % input_name)
        self.model_weight_init.init_weight(result)
        return result

    def get_model_from_cfg(self, cfg_path, default_args=None):
        if not cfg_path.endswith("cfg"):
            EasyLogger.error("%s model error" % cfg_path)
            return None
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.modelParse.readCfgFile(cfg_path)
        model = MyModel(model_define, path, default_args)
        model.set_name(file_name)
        return model

    def get_model_from_name(self, model_config):
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        model_name = model_config['type'].strip()
        model_config['type'] = model_name
        EasyLogger.debug(model_config)
        if REGISTERED_CLS_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_CLS_MODEL)
        elif REGISTERED_DET2D_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_DET2D_MODEL)
        elif REGISTERED_SEG_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_SEG_MODEL)
        elif REGISTERED_SR_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_SR_MODEL)
        elif REGISTERED_GAN_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_GAN_MODEL)
        elif REGISTERED_KEYPOINT2D_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_KEYPOINT2D_MODEL)
        elif REGISTERED_RNN_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_RNN_MODEL)
        elif REGISTERED_REID_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_REID_MODEL)
        elif REGISTERED_MULTI_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_MULTI_MODEL)
        elif REGISTERED_PC_CLS_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_PC_CLS_MODEL)
        elif REGISTERED_PC_SEG_MODEL.has_class(model_name):
            model = build_from_cfg(model_config, REGISTERED_PC_SEG_MODEL)
        else:
            model = None
        return model
