#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.utility.logger import EasyLogger

try:
    from easyai import _C
    try:
        from apex import amp
        # Only valid with fp32 inputs - give AMP the hint
        nms = amp.float_function(_C.nms)
    except ImportError:
        nms = _C.nms
        EasyLogger.error("import amp fail!")
except ImportError:
    nms = None
    EasyLogger.error("import _C fail!")

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
