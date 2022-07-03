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
from easyai.model_block.head.det2d.yolo_box_head import YoloBoxHead
from easyai.model_block.base_block.common.utility_layer import RouteLayer
from easyai.model_block.base_block.common.utility_block import ConvBNActivationBlock

from easyai.model.utility.base_det_model import *
from easyai.model.utility.model_registry import REGISTERED_DET2D_MODEL


@REGISTERED_DET2D_MODEL.register_module(ModelName.YoloV5sDet2d)
class YoloV5sDet2d(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=80):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.YoloV5sDet2d)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.Swish

        self.model_args['type'] = BackboneName.Yolov5s_Backbone

        self.anchor_count = 3
        self.anchor_sizes = "10,13|16,30|33,23|" \
                            "30,61|62,45|59,119|" \
                            "116,90|156,198|373,326"
        self.loss_config = {"type": LossName.YoloV5Loss,
                            "anchor_count": self.anchor_count,
                            "anchor_sizes": self.anchor_sizes,
                            "class_number": class_number,
                            "box_weight": 0.05,
                            "object_weight": 1.0,
                            "class_weight": 0.5}

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

        layer6 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=128,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer6.get_name(), layer6, 128)

        route6 = RouteLayer('-1,-5')
        output_channel = route6.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route6.get_name(), route6, output_channel)

        layer7 = C3Block(in_channels=self.block_out_channels[-1],
                         out_channels=256,
                         number=1,
                         shortcut=False,
                         groups=1,
                         bnName=self.bn_name,
                         activationName=self.activation_name,
                         expansion=0.5)
        self.add_block_list(layer7.get_name(), layer7, 256)

        layer9 = ConvBNActivationBlock(in_channels=self.block_out_channels[-1],
                                       out_channels=256,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer9.get_name(), layer9, 256)

        route8 = RouteLayer('-1,-12')
        output_channel = route8.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route8.get_name(), route8, output_channel)

        layer10 = C3Block(in_channels=self.block_out_channels[-1],
                          out_channels=512,
                          number=1,
                          shortcut=False,
                          groups=1,
                          bnName=self.bn_name,
                          activationName=self.activation_name,
                          expansion=0.5)
        self.add_block_list(layer10.get_name(), layer10, 512)

        head = YoloBoxHead("-7,-4,-1", (128, 256, 512), self.class_number,
                           self.anchor_count)
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
            elif HeadType.YoloBoxHead in key:
                x = block(layer_outputs, base_outputs)
            elif self.loss_factory.has_loss(key):
                output.extend(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
