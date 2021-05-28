#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.model.utility.abstract_model import *
from easyai.model_block.backbone.utility import BackboneFactory
from easyai.loss.utility.loss_factory import LossFactory


class BaseGanModel(AbstractModel):

    def __init__(self, data_channel):
        super().__init__()
        self.d_model_list = []
        self.g_model_list = []
        self.d_loss_list = []
        self.g_loss_list = []
        self.data_channel = data_channel

        self.gan_base_factory = BackboneFactory()
        self.loss_factory = LossFactory()

    def clear_loss(self):
        self.d_loss_list = []
        self.g_loss_list = []

    @abc.abstractmethod
    def create_loss_list(self, input_dict=None):
        pass
