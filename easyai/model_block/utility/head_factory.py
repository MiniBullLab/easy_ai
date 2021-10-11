#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
import os.path
from easyai.model_block.utility.block_registry import REGISTERED_MODEL_HEAD
from easyai.utility.registry import build_from_cfg

__all__ = ["HeadFactory"]


class HeadFactory():

    def __init__(self):
        pass

    def get_model_head(self, head_config):
        result = None
        EasyLogger.debug(head_config)
        try:
            input_name = head_config['type'].strip()
            head_args = head_config.copy()
            if REGISTERED_MODEL_HEAD.has_class(input_name):
                result = build_from_cfg(head_args, REGISTERED_MODEL_HEAD)
            if result is None:
                EasyLogger.error("head:%s error" % input_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result
