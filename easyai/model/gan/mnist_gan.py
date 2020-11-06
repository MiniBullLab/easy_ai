#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import BlockType
from easyai.base_name.loss_name import LossName
from easyai.base_name.backbone_name import GanBaseModelName
from easyai.model.utility.base_gan_model import *
from easyai.model.utility.registry import REGISTERED_GAN_MODEL


@REGISTERED_GAN_MODEL.register_module(ModelName.MNISTGan)
class MNISTGan(BaseGanModel):

    def __init__(self, data_channel=1, image_size=(28, 28)):
        super().__init__(data_channel)
        self.set_name(ModelName.MNISTGan)
        self.image_size = image_size
        self.z_dimension = 100  # the dimension of noise tensor

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.d_model_args['type'] = GanBaseModelName.MNISTDiscriminator
        discriminator = self.gan_base_factory.get_backbone_model(self.model_args)
        d_out_channels = discriminator.get_outchannel_list()
        self.add_block_list(BlockType.Discriminator, discriminator, d_out_channels[-1], 1)

        self.g_model_args['type'] = GanBaseModelName.MNISTGenerator
        self.g_model_args['final_out_channel'] = self.image_size[0] * self.image_size[1]
        generator = self.gan_base_factory.get_backbone_model(self.model_args)
        g_out_channels = generator.get_outchannel_list()
        self.add_block_list(BlockType.Generator, generator, g_out_channels[-1], 1)

    def create_loss_list(self, input_dict=None):
        self.clear_loss()
        loss_config = {"type": LossName.MNISTDiscriminatorLoss}
        loss1 = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss1.get_name(), loss1, self.block_out_channels[-1], 1)
        self.d_loss_list.append(loss1)

        loss_config = {"type": LossName.MNISTGeneratorLoss}
        loss2 = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss2.get_name(), loss2, self.block_out_channels[-1], 1)
        self.g_loss_list.append(loss2)

    def generator_input_data(self, inputs_data, data_type=0):
        result = None
        if data_type == 0:
            result = torch.flatten(inputs_data, start_dim=1)
        elif data_type == 1:
            # Random noise from N(0,1)
            result = torch.randn((inputs_data.size(0), self.z_dimension))
        return result

    def forward(self, fake_data, real_data=None, net_type=0):
        output = []
        if net_type == 0:
            x = self._modules[BlockType.Discriminator](real_data)
            output.append(x)
            fake_x = self._modules[BlockType.Generator](fake_data)
            x = self._modules[BlockType.Discriminator](fake_x.detach())
            output.append(x)
        elif net_type == 1:
            x = self._modules[BlockType.Generator](fake_data)
            x = self._modules[BlockType.Discriminator](x)
            output.append(x)
        elif net_type == 3:
            x = self._modules[BlockType.Generator](fake_data)
            output.append(x)
        return output
