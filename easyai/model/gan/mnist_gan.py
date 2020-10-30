#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:

from easyai.base_name.model_name import ModelName
from easyai.base_name.block_name import ActivationType
from easyai.base_name.block_name import BlockType
from easyai.base_name.loss_name import LossType
from easyai.base_name.backbone_name import BackboneName
from easyai.model.utility.base_model import *
from easyai.model.backbone.utility.backbone_factory import BackboneFactory
from easyai.model.base_block.utility.utility_block import FcActivationBlock
from easyai.loss.gan.mnist_discriminator_loss import MNISTDiscriminatorLoss
from easyai.loss.gan.mnist_generator_loss import MNISTGeneratorLoss
from easyai.model.utility.registry import REGISTERED_GAN_MODEL


@REGISTERED_GAN_MODEL.register_module(ModelName.ClassNet)
class MNISTGan(BaseModel):

    def __init__(self, data_channel=1, image_size=(28, 28)):
        super().__init__(data_channel)
        self.image_size = image_size
        self.z_dimension = 100  # the dimension of noise tensor
        self.factory = BackboneFactory()
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        self.create_loss()

        self.model_args['type'] = BackboneName.MNISTDiscriminator
        discriminator = self.factory.get_backbone_model(self.model_args)
        d_out_channels = discriminator.get_outchannel_list()
        self.add_block_list(BlockType.Discriminator, discriminator, d_out_channels[-1])

        self.model_args['type'] = BackboneName.MNISTGenerator
        generator = self.factory.get_backbone_model(self.model_args)
        g_out_channels = generator.get_outchannel_list()
        self.add_block_list(BlockType.Generator, generator, g_out_channels[-1])

        out_channel = self.image_size[0] * self.image_size[1]
        layer1 = FcActivationBlock(g_out_channels[-1], out_channel,
                                   activationName=ActivationType.Tanh)
        self.add_block_list(layer1.get_name(), layer1, out_channel)

        self.model_args['type'] = BackboneName.MNISTDiscriminator
        discriminator = self.factory.get_backbone_model(self.model_args)
        d_out_channels = discriminator.get_outchannel_list()
        self.add_block_list(BlockType.Discriminator, discriminator, d_out_channels[-1])

        loss1 = MNISTDiscriminatorLoss()
        self.add_block_list(loss1.get_name(), loss1, self.block_out_channels[-1])
        self.lossList.append(loss1)

        self.model_args['type'] = BackboneName.MNISTGenerator
        generator = self.factory.get_backbone_model(self.model_args)
        g_out_channels = generator.get_outchannel_list()
        self.add_block_list(BlockType.EndGanNet, generator, g_out_channels[-1])

        out_channel = self.image_size[0] * self.image_size[1]
        layer2 = FcActivationBlock(g_out_channels[-1], out_channel,
                                   activationName=ActivationType.Tanh)
        self.add_block_list(layer2.get_name(), layer2, out_channel)

        loss2 = MNISTGeneratorLoss()
        self.add_block_list(loss2.get_name(), loss2, self.block_out_channels[-1])
        self.lossList.append(loss2)

    def create_loss(self, input_dict=None):
        self.lossList = []

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
        x = None
        for key, block in self._modules.items():
            if net_type == 0:
                if BlockType.EndGanNet not in key:
                    continue
                elif BlockType.EndGanNet in key:
                    x = block(fake_data)
                elif LossType.MNISTGeneratorLoss in key:
                    output.append(x)
                else:
                    x = block(x)
                print(key, x.shape)
            elif net_type == 1:
                x = real_data
                if BlockType.Discriminator in key:
                    x = block(x)
                    output.append(x)
                elif BlockType.Generator in key:
                    x = block(fake_data)
                elif LossType.MNISTDiscriminatorLoss in key:
                    pass
                else:
                    x = block(x)
                print(key, x.shape)
        return output
