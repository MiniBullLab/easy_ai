#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

import abc


class BaseEvaluation():

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
