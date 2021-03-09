#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:


from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import BlockType, HeadType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.fpn_block import FPNBlock
from easyai.model.model_block.head.rpn_head import MultiRPNHead
from easyai.model.utility.base_det_model import *
from easyai.model.utility.registry import REGISTERED_DET2D_MODEL


@REGISTERED_DET2D_MODEL.register_module(ModelName.RPNDet2d)
class RPNDet2d(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, 1)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU

        self.model_args['type'] = BackboneName.ResNet50

        self.loss_config = {"type": LossName.RPNLoss,
                            "input_size": "640,640",
                            "anchor_sizes": "32,64,128,256,512",
                            "aspect_ratios": "0.5,1.0,2.0",
                            "anchor_strides": "4,8,16,32,64",
                            "fg_iou_threshold": 0.5,
                            "bg_iou_threshold": 0.5,
                            "per_image_sample": 256,
                            "positive_fraction": 0.5}
        self.anchor_number = 3

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        down_layers = [4, 8, 14, 17]
        down_layer_outputs = [self.block_out_channels[i] if i < 0 else base_out_channels[i]
                              for i in down_layers]
        temp_str = ",".join('%s' % index for index in down_layers)
        fpn_layer = FPNBlock(temp_str, down_layer_outputs, 256)
        self.add_block_list(fpn_layer.get_name(), fpn_layer, 256)

        head_layer = MultiRPNHead(256, self.anchor_number,
                                  activation_name=self.activation_name)
        self.add_block_list(head_layer.get_name(), head_layer, 256)

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss = self.loss_factory.get_loss(self.loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        multi_output = []
        for key, block in self._modules.items():
            if BlockType.BaseNet in key:
                base_outputs = block(x)
                x = base_outputs[-1]
            elif BlockType.FPNBlock in key:
                x = block(layer_outputs, base_outputs)
                multi_output.extend(x)
            elif HeadType.MultiRPNHead in key:
                x = block(x)
                multi_output.extend(x)
            elif self.loss_factory.has_loss(key):
                temp_output = self.loss_factory.get_loss_input(key, x, multi_output)
                output.extend(temp_output)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
