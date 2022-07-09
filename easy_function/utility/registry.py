#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import inspect
import six
from easyai.utility.logger import EasyLogger


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get_keys(self):
        return self._module_dict.keys()

    def get(self, key):
        return self._module_dict.get(key, None)

    def has_class(self, key):
        result = key.strip() in self._module_dict
        return result

    def add_module(self, module_class, cls_name=None):
        if not inspect.isclass(module_class):
            raise TypeError(
                "module must be a class, but got {}".format(type(module_class))
            )
        if cls_name is None:
            cls_name = module_class.__name__
        if cls_name in self._module_dict:
            raise KeyError(
                "{} is already registered in {}".format(cls_name, self.name)
            )
        self._module_dict[cls_name] = module_class
        # print(module_class, "register name: %s" % cls_name)

    def _register_module(self, module_class, cls_name=None):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.

        """
        if not inspect.isclass(module_class):
            raise TypeError(
                "module must be a class, but got {}".format(type(module_class))
            )
        if cls_name is None:
            cls_name = module_class.__name__
        # print(self._module_dict)
        if cls_name in self._module_dict:
            raise KeyError(
                "{} is already registered in {}".format(cls_name, self.name)
            )
        self._module_dict[cls_name] = module_class
        EasyLogger.debug("{} register name: {}".format(module_class, cls_name))

    def register_module(self, cls_name=None):
        def deco(cls):
            self._register_module(cls, cls_name)
            return cls
        return deco


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if isinstance(obj_type, six.string_types):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, registry.name)
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a ocr or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)