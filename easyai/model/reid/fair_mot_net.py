#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.backbone_name import BackboneName
from easyai.name_manager.block_name import NormalizationType, ActivationType
from easyai.name_manager.block_name import LayerType, BlockType, HeadType
from easyai.name_manager.loss_name import LossName
from easyai.model_block.base_block.common.upsample_layer import Upsample
from easyai.model_block.base_block.det2d.yolov5_block import C3Block
from easyai.model_block.base_block.common.utility_layer import RouteLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock
from easyai.model_block.head.reid.fair_mot_head import FairMOTHead
from easyai.model.utility.base_reid_model import BaseReIDModel
from easyai.model.utility.model_registry import REGISTERED_REID_MODEL


@REGISTERED_REID_MODEL.register_module(ModelName.FairMOTNet)
class FairMOTNet(BaseReIDModel):

    def __init__(self, data_channel=3, class_number=1,
                 reid=64, max_id=-1,
                 backbone_name=BackboneName.Yolov5s_Old_Backbone):
        super().__init__(data_channel, class_number, reid, max_id)
        self.set_name(ModelName.FairMOTNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.Swish

        self.model_args['type'] = backbone_name

        if max_id > 0:
            self.loss_config = {"type": LossName.FairMotLoss,
                                "class_number": class_number,
                                "reid": reid,
                                "max_id": max_id}
        else:
            self.loss_config = {"type": LossName.EmptyLoss}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = ConvBNActivationBlock(in_channels=base_out_channels[-1],
                                       out_channels=256,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, 256)

        up1 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(up1.get_name(), up1, self.block_out_channels[-1])

        route2 = RouteLayer('-1,6')
        output_channel = route2.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route2.get_name(), route2, output_channel)

        layer2 = C3Block(in_channels=self.block_out_channels[-1],
                         out_channels=256,
                         number=1,
                         shortcut=False,
                         groups=1,
                         bnName=self.bn_name,
                         activationName=self.activation_name,
                         expansion=0.5)
        self.add_block_list(layer2.get_name(), layer2, 256)

        layer3 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=128,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, 128)

        up3 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(up3.get_name(), up3, self.block_out_channels[-1])

        route4 = RouteLayer('-1,4')
        output_channel = route4.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route4.get_name(), route4, output_channel)

        layer4 = C3Block(in_channels=self.block_out_channels[-1],
                         out_channels=128,
                         number=1,
                         shortcut=False,
                         groups=1,
                         bnName=self.bn_name,
                         activationName=self.activation_name,
                         expansion=0.5)
        self.add_block_list(layer4.get_name(), layer4, 128)

        layer5 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=64,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer5.get_name(), layer5, 64)

        up4 = Upsample(scale_factor=2, mode='nearest')
        self.add_block_list(up4.get_name(), up4, self.block_out_channels[-1])

        route5 = RouteLayer('-1,2')
        output_channel = route5.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route5.get_name(), route5, output_channel)

        layer6 = C3Block(in_channels=self.block_out_channels[-1],
                         out_channels=64,
                         number=1,
                         shortcut=False,
                         groups=1,
                         bnName=self.bn_name,
                         activationName=self.activation_name,
                         expansion=0.5)
        self.add_block_list(layer6.get_name(), layer6, 64)

        head = FairMOTHead(64, (self.class_number, self.reid, 2, 4),
                           self.activation_name)
        self.add_block_list(head.get_name(), head, self.class_number)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
        self.add_block_list(loss.get_name(), loss, -1)
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif LayerType.RouteLayer in key:
                x = block(layer_outputs, base_outputs)
            elif LayerType.ShortcutLayer in key:
                x = block(layer_outputs)
            elif HeadType.FairMOTHead in key:
                x = block(x)
            elif self.loss_factory.has_loss(key):
                output.extend(x)
            else:
                x = block(x)
                # print(key, x.shape)
            layer_outputs.append(x)
        return output
