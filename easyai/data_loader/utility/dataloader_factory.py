#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET
from easyai.data_loader.utility.dataloader_registry import REGISTERED_DATASET_COLLATE
from easyai.data_loader.utility.dataloader_registry import REGISTERED_TRAIN_DATALOADER
from easyai.data_loader.utility.dataloader_registry import REGISTERED_VAL_DATALOADER
from easyai.utility.registry import build_from_cfg


class DataloaderFactory():

    def __init__(self):
        pass

    def get_train_dataloader(self, dataloader_config, dataset_config):
        result = None
        if dataset_config is not None:
            type_name = dataset_config['type'].strip()
            if REGISTERED_DATASET.has_class(type_name):
                dataset = build_from_cfg(dataset_config, REGISTERED_DATASET)
                type_name = dataloader_config['type'].strip()
                if REGISTERED_TRAIN_DATALOADER.has_class(type_name):
                    config_args = self.get_collate_fn(dataloader_config)
                    config_args['dataset'] = dataset
                    result = build_from_cfg(config_args, REGISTERED_TRAIN_DATALOADER)
                else:
                    print("%s dataloader not exits" % type_name)

            else:
                print("%s dataset not exits" % type_name)
        else:
            type_name = dataloader_config['type'].strip()
            if REGISTERED_TRAIN_DATALOADER.has_class(type_name):
                result = build_from_cfg(dataloader_config, REGISTERED_TRAIN_DATALOADER)
            else:
                print("%s dataloader not exits" % type_name)
        return result

    def get_val_dataloader(self, dataloader_config, dataset_config):
        result = None
        if dataset_config is not None:
            type_name = dataset_config['type'].strip()
            if REGISTERED_DATASET.has_class(type_name):
                dataset = build_from_cfg(dataset_config, REGISTERED_DATASET)
                type_name = dataloader_config['type'].strip()
                if REGISTERED_VAL_DATALOADER.has_class(type_name):
                    config_args = self.get_collate_fn(dataloader_config)
                    config_args['dataset'] = dataset
                    result = build_from_cfg(config_args, REGISTERED_VAL_DATALOADER)
                else:
                    print("%s dataloader not exits" % type_name)

            else:
                print("%s dataset not exits" % type_name)
        else:
            type_name = dataloader_config['type'].strip()
            if REGISTERED_VAL_DATALOADER.has_class(type_name):
                result = build_from_cfg(dataloader_config, REGISTERED_VAL_DATALOADER)
            else:
                print("%s dataloader not exits" % type_name)
        return result

    def get_collate_fn(self, dataloader_config):
        config_args = dataloader_config.copy()
        if 'collate_fn' in config_args:
            type_name = config_args['collate_fn']['type'].strip()
            if REGISTERED_DATASET_COLLATE.has_class(type_name):
                collate_fn = build_from_cfg(config_args.pop('collate_fn'),
                                            REGISTERED_DATASET_COLLATE)
                config_args['collate_fn'] = collate_fn
            else:
                print("%s collate_fn not exits" % type_name)
                config_args.pop('collate_fn')
        return config_args
