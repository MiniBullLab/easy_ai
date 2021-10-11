#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie


class BasePostProcess():

    def __init__(self):
        self.threshold = 0

    def set_threshold(self, value):
        self.threshold = value
