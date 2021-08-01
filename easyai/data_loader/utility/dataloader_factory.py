#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import traceback
from easyai.utility.logger import EasyLogger
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE
from easyai.data_loader.utility.dataloader_registry import REGISTERED_TRAIN_DATALOADER
from easyai.data_loader.utility.dataloader_registry import REGISTERED_VAL_DATALOADER
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATA_TRANSFORMS
from easyai.data_loader.utility.data_transforms_factory import DataTransformsFactory
from easyai.utility.registry import build_from_cfg


class DataloaderFactory():

    def __init__(self):
        self.transforms_factory = DataTransformsFactory()

    def get_train_dataloader(self, train_path, dataloader_config, dataset_config):
        result = None
        EasyLogger.debug(train_path)
        EasyLogger.debug(dataloader_config)
        EasyLogger.debug(dataset_config)
        try:
            if dataset_config is not None:
                type_name = dataset_config['type'].strip()
                copy_dataset_config = dataset_config.copy()
                copy_dataset_config['data_path'] = train_path
                if REGISTERED_DATASET.has_class(type_name):
                    copy_dataset_config = self.get_transform_fn(copy_dataset_config)
                    dataset = build_from_cfg(copy_dataset_config, REGISTERED_DATASET)
                    type_name = dataloader_config['type'].strip()
                    if REGISTERED_TRAIN_DATALOADER.has_class(type_name):
                        config_args = self.get_collate_fn(dataloader_config)
                        config_args['dataset'] = dataset
                        result = build_from_cfg(config_args, REGISTERED_TRAIN_DATALOADER)
                    else:
                        EasyLogger.error("%s dataloader not exits" % type_name)
                else:
                    EasyLogger.error("%s dataset not exits" % type_name)
            else:
                type_name = dataloader_config['type'].strip()
                config_args = dataloader_config.copy()
                config_args['data_path'] = train_path
                if REGISTERED_TRAIN_DATALOADER.has_class(type_name):
                    config_args = self.get_transform_fn(config_args)
                    result = build_from_cfg(config_args, REGISTERED_TRAIN_DATALOADER)
                else:
                    EasyLogger.error("%s dataloader not exits" % type_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result

    def get_val_dataloader(self, val_path, dataloader_config, dataset_config):
        result = None
        EasyLogger.debug(val_path)
        EasyLogger.debug(dataloader_config)
        EasyLogger.debug(dataset_config)
        try:
            if dataset_config is not None and len(dataset_config) > 0:
                type_name = dataset_config['type'].strip()
                copy_dataset_config = dataset_config.copy()
                copy_dataset_config['data_path'] = val_path
                if REGISTERED_DATASET.has_class(type_name):
                    copy_dataset_config = self.get_transform_fn(copy_dataset_config)
                    dataset = build_from_cfg(copy_dataset_config, REGISTERED_DATASET)
                    type_name = dataloader_config['type'].strip()
                    if REGISTERED_VAL_DATALOADER.has_class(type_name):
                        config_args = self.get_collate_fn(dataloader_config)
                        config_args['dataset'] = dataset
                        result = build_from_cfg(config_args, REGISTERED_VAL_DATALOADER)
                    else:
                        EasyLogger.error("%s dataloader not exits" % type_name)
                else:
                    EasyLogger.error("%s dataset not exits" % type_name)
            else:
                type_name = dataloader_config['type'].strip()
                config_args = dataloader_config.copy()
                config_args['data_path'] = val_path
                if REGISTERED_VAL_DATALOADER.has_class(type_name):
                    config_args = self.get_transform_fn(config_args)
                    result = build_from_cfg(config_args, REGISTERED_VAL_DATALOADER)
                else:
                    EasyLogger.error("%s dataloader not exits" % type_name)
        except ValueError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        except TypeError as err:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(err)
        return result

    def get_collate_fn(self, dataloader_config):
        config_args = None
        try:
            config_args = dataloader_config.copy()
            if 'collate_fn' in config_args:
                type_name = config_args['collate_fn']['type'].strip()
                if REGISTERED_DATASET_COLLATE.has_class(type_name):
                    collate_fn = build_from_cfg(config_args.pop('collate_fn'),
                                                REGISTERED_DATASET_COLLATE)
                    config_args['collate_fn'] = collate_fn
                else:
                    EasyLogger.error("%s collate_fn not exits" % type_name)
                    config_args.pop('collate_fn')
        except ValueError:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(config_args)
        except TypeError:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(config_args)
        return config_args

    def get_transform_fn(self, dataset_config):
        config_args = None
        try:
            config_args = dataset_config.copy()
            if 'transform_func' in config_args:
                type_name = config_args['transform_func']['type'].strip()
                if REGISTERED_DATA_TRANSFORMS.has_class(type_name):
                    transform_func = build_from_cfg(config_args.pop('collate_fn'),
                                                    REGISTERED_DATA_TRANSFORMS)
                    config_args['transform_func'] = transform_func
                else:
                    EasyLogger.error("%s transform_func not exits" % type_name)
                    config_args.pop('transform_func')
        except ValueError:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(config_args)
        except TypeError:
            EasyLogger.error(traceback.format_exc())
            EasyLogger.error(config_args)
        return config_args
