#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
import os.path
from easyai.model_block.backbone.common.my_backbone import MyBackbone
from easyai.model_block.utility.model_parse import ModelParse
from easyai.model_block.utility.block_registry import REGISTERED_VISION_BACKBONE
from easyai.model_block.utility.block_registry import REGISTERED_CLS_BACKBONE
from easyai.model_block.utility.block_registry import REGISTERED_GAN_D_BACKBONE
from easyai.model_block.utility.block_registry import REGISTERED_GAN_G_BACKBONE
from easyai.model_block.utility.block_registry import REGISTERED_PC_CLS_BACKBONE
from easyai.utility.registry import build_from_cfg

__all__ = ["BackboneFactory"]


class BackboneFactory():

    def __init__(self):
        self.cfgReader = ModelParse()

    def get_backbone_model(self, model_config):
        result = None
        EasyLogger.debug(model_config)
        try:
            input_name = model_config['type'].strip()
            model_args = model_config.copy()
            if input_name.endswith("cfg"):
                result = self.get_backbone_from_cfg(input_name)
            else:
                result = self.get_backbone_from_name(model_args)
                if result is None:
                    result = self.get_troch_vision_model(model_config)
                if result is None:
                    result = self.get_gan_base_model(model_config)
                if result is None:
                    result = self.get_pc_backbone_from_name(model_config)
                if result is None:
                    EasyLogger.error("backbone:%s error" % input_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result

    def get_backbone_from_cfg(self, cfg_path):
        path, file_name_and_post = os.path.split(cfg_path)
        file_name, post = os.path.splitext(file_name_and_post)
        model_define = self.cfgReader.readCfgFile(cfg_path)
        model = MyBackbone(model_define)
        model.set_name(file_name)
        return model

    def get_backbone_from_name(self, model_config):
        input_name = model_config['type'].strip()
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        result = None
        if REGISTERED_CLS_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_CLS_BACKBONE)
        return result

    def get_troch_vision_model(self, model_config):
        input_name = model_config['type'].strip()
        if model_config.get('pretrained') is None:
            model_config['pretrained'] = True
        result = None
        if REGISTERED_VISION_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_VISION_BACKBONE)
        return result

    def get_gan_base_model(self, model_config):
        input_name = model_config['type'].strip()
        if model_config.get('data_channel') is None:
            model_config['data_channel'] = 3
        result = None
        if REGISTERED_GAN_D_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_GAN_D_BACKBONE)
        elif REGISTERED_GAN_G_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_GAN_G_BACKBONE)
        return result

    def get_pc_backbone_from_name(self, model_config):
        input_name = model_config['type'].strip()
        result = None
        if REGISTERED_PC_CLS_BACKBONE.has_class(input_name):
            result = build_from_cfg(model_config, REGISTERED_PC_CLS_BACKBONE)
        return result
