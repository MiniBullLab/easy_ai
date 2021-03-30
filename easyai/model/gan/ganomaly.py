#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import BlockType
from easyai.base_name.loss_name import LossName
from easyai.base_name.backbone_name import GanBaseModelName
from easyai.model.utility.base_gan_model import *
from easyai.model.utility.registry import REGISTERED_GAN_MODEL


@REGISTERED_GAN_MODEL.register_module(ModelName.GANomaly)
class GANomaly(BaseGanModel):

    def __init__(self, data_channel=3, image_size=(32, 32)):
        super().__init__(data_channel)
        self.set_name(ModelName.GANomaly)
        self.image_size = image_size
        self.z_dimension = 100  # the dimension of noise tensor

        self.create_block_list()

        assert len(self.d_model_list) == len(self.d_loss_list)
        assert len(self.g_model_list) == len(self.g_loss_list)

    def create_block_list(self):
        self.clear_list()
        self.d_model_list = []
        self.g_model_list = []

        d_model_args = dict()
        d_model_args['type'] = GanBaseModelName.GANomalyDiscriminator
        d_model_args["data_channel"] = self.data_channel
        d_model_args["input_size"] = self.image_size[0]
        discriminator = self.gan_base_factory.get_backbone_model(d_model_args)
        d_out_channels = discriminator.get_outchannel_list()
        self.add_block_list(BlockType.Discriminator, discriminator, d_out_channels[-1], 1)
        self.d_model_list.append(discriminator)

        g_model_args = dict()
        g_model_args['type'] = GanBaseModelName.GANomalyGenerator
        g_model_args["data_channel"] = self.data_channel
        g_model_args["input_size"] = self.image_size[0]
        g_model_args['final_out_channel'] = self.z_dimension
        generator = self.gan_base_factory.get_backbone_model(g_model_args)
        g_out_channels = generator.get_outchannel_list()
        self.add_block_list(BlockType.Generator, generator, g_out_channels[-1], 1)
        self.g_model_list.append(generator)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.clear_loss()
        loss_config = {"type": LossName.GANomalyDiscriminatorLoss}
        loss1 = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss1.get_name(), loss1, self.block_out_channels[-1], 1)
        self.d_loss_list.append(loss1)

        loss_config = {"type": LossName.GANomalyGeneratorLoss}
        loss2 = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss2.get_name(), loss2, self.block_out_channels[-1], 1)
        self.g_loss_list.append(loss2)

    def forward(self, real_data, fake_data=None, net_type=0):
        output = []
        if net_type == 0:
            x = self._modules[BlockType.Generator](real_data)
            output.extend(x)
        elif net_type == 1:
            output.append(real_data)
            x = self._modules[BlockType.Generator](real_data)
            output.extend(x)
            d_x = self._modules[BlockType.Discriminator](real_data)
            output.append(d_x[0])
            d_x = self._modules[BlockType.Discriminator](x[1])
            output.append(d_x[0])
        elif net_type == 2:
            x = self._modules[BlockType.Discriminator](real_data)
            output.append(x[-1])
            x = self._modules[BlockType.Discriminator](fake_data.detach())
            output.append(x[-1])
        return output
