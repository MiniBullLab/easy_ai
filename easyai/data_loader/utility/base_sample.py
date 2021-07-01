#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.helper.dir_process import DirProcess
from easyai.helper.json_process import JsonProcess


class BaseSample():

    def __init__(self):
        self.dirProcess = DirProcess()
        self.json_process = JsonProcess()