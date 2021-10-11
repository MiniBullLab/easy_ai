#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATA_TRANSFORMS
from easyai.utility.registry import build_from_cfg


class DataTransformsFactory():

    def __init__(self):
        pass

    def get_data_transform(self, data_transform_config):
        result = None
        EasyLogger.debug(data_transform_config)
        if data_transform_config is None:
            return result
        try:
            type_name = data_transform_config['type'].strip()
            if REGISTERED_DATA_TRANSFORMS.has_class(type_name):
                result = build_from_cfg(data_transform_config, REGISTERED_DATA_TRANSFORMS)
            else:
                EasyLogger.error("%s data transform not exits" % type_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result
