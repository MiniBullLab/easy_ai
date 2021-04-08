#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.model_name import ModelName
from easyai.base_name.backbone_name import BackboneName
from easyai.base_name.block_name import NormalizationType, ActivationType
from easyai.base_name.block_name import LayerType, BlockType
from easyai.base_name.loss_name import LossName
from easyai.model.model_block.base_block.utility.utility_layer import RouteLayer
from easyai.model.model_block.base_block.utility.fpn_block import FPNV2Block
from easyai.model.model_block.base_block.utility.utility_block import ConvBNActivationBlock
from easyai.model.model_block.head.ssd_box_head import SSDBoxHead, MultiSSDBoxHead
from easyai.model.utility.base_det_model import *
from easyai.model.utility.registry import REGISTERED_DET2D_MODEL


class MobileV2RefineDet2d(BaseDetectionModel):

    def __init__(self, data_channel=3, class_number=1):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.MobileV2RefineDet2d)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.feature_out_channels = 64

        self.model_args['type'] = BackboneName.MobileNetV2_0_5

        self.anchor_sizes = "10,10|24,26|33,68|" \
                            "61,131|70,36|117,78|" \
                            "128,209|277,107|311,251"
        self.loss_config = {"type": LossName.YoloV3Loss,
                            "anchor_sizes": self.anchor_sizes,
                            "class_number": class_number,
                            "anchor_mask": "6,7,8",
                            "reduction": 32,
                            "coord_weight": 3.0,
                            "noobject_weight": 1.0,
                            "object_weight": 1.0,
                            "class_weight": 1.0,
                            "iou_threshold": 0.5}

        self.create_block_list()

    def create_block_list(self):
        self.clear_list()
        self.create_loss_list()

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        layer1 = ConvBNActivationBlock(in_channels=base_out_channels[-1],
                                       out_channels=self.feature_out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer1.get_name(), layer1, self.feature_out_channels)

        route1 = RouteLayer("6")
        output_channel1 = route1.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route1.get_name(), route1, output_channel1)

        arm_head1 = SSDBoxHead(output_channel1, 9, 2, is_gaussian=False)
        self.add_block_list(arm_head1.get_name(), arm_head1, 2)

        route2 = RouteLayer("13")
        output_channel2 = route2.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route2.get_name(), route2, output_channel2)

        arm_head2 = SSDBoxHead(output_channel2, 3, 2, is_gaussian=False)
        self.add_block_list(arm_head2.get_name(), arm_head2, 2)

        route3 = RouteLayer("17")
        output_channel3 = route3.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route3.get_name(), route3, output_channel3)

        arm_head3 = SSDBoxHead(output_channel3, 3, 2, is_gaussian=False)
        self.add_block_list(arm_head3.get_name(), arm_head3, 2)

        route4 = RouteLayer("-7")
        output_channel4 = route4.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route4.get_name(), route4, output_channel4)

        layer2 = ConvBNActivationBlock(in_channels=self.feature_out_channels,
                                       out_channels=self.feature_out_channels,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer2.get_name(), layer2, self.feature_out_channels)

        arm_head4 = SSDBoxHead(self.feature_out_channels, 3, 2, is_gaussian=False)
        self.add_block_list(arm_head4.get_name(), arm_head4, 2)

        down_layers = [6, 13, 17]
        down_layer_outputs = [self.block_out_channels[i] if i < 0 else base_out_channels[i]
                              for i in down_layers]
        temp_str = ",".join('%s' % index for index in down_layers)
        fpn_layer = FPNV2Block(temp_str, down_layer_outputs, self.feature_out_channels)
        self.add_block_list(fpn_layer.get_name(), fpn_layer, self.feature_out_channels)

        input_channels = [self.feature_out_channels] * 3
        anchor_numbers = [9, 3, 3]
        odm_head3 = MultiSSDBoxHead(input_channels, anchor_numbers,
                                    self.class_number, is_gaussian=True)
        self.add_block_list(odm_head3.get_name(), odm_head3, self.class_number)

        route5 = RouteLayer("-4")
        output_channel5 = route5.get_output_channel(base_out_channels, self.block_out_channels)
        self.add_block_list(route5.get_name(), route5, output_channel5)

        layer3 = ConvBNActivationBlock(in_channels=self.feature_out_channels,
                                       out_channels=self.feature_out_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bnName=self.bn_name,
                                       activationName=self.activation_name)
        self.add_block_list(layer3.get_name(), layer3, self.feature_out_channels)

        odm_head4 = SSDBoxHead(self.feature_out_channels, 3, self.class_number,
                               is_gaussian=True)
        self.add_block_list(odm_head4.get_name(), odm_head4, self.class_number)

    def create_loss_list(self, input_dict=None):
        self.lossList = []

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
            elif self.loss_factory.has_loss(key):
                output.append(x)
            else:
                x = block(x)
            # print(key, x.shape)
            layer_outputs.append(x)
        return output
