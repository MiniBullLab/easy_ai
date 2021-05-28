#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:
"""BiSeNet
    Reference:
        Changqian Yu, et al. "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation"
        arXiv preprint arXiv:1808.00897 (2018).
"""

from easyai.config.name_manager import ModelName
from easyai.config.name_manager import BackboneName
from easyai.config.name_manager import NormalizationType, ActivationType
from easyai.config.name_manager import LayerType, BlockType
from easyai.config.name_manager import LossName
from easyai.model_block.base_block.utility.upsample_layer import Upsample
from easyai.model_block.base_block.utility.utility_layer import RouteLayer
from easyai.model_block.base_block import ConvBNActivationBlock
from easyai.model_block.base_block.seg.bisenet_block import BiSeNetBlockName
from easyai.model_block.base_block.seg.bisenet_block import SpatialPath, GlobalAvgPooling
from easyai.model_block.base_block.seg.bisenet_block import ContextPath, FeatureFusionBlock
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.registry import REGISTERED_SEG_MODEL


@REGISTERED_SEG_MODEL.register_module(ModelName.BiSeNet)
class BiSeNet(BaseClassifyModel):

    def __init__(self, data_channel=3, class_number=2):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.BiSeNet)
        self.bn_name = NormalizationType.BatchNormalize2d
        self.activation_name = ActivationType.ReLU
        self.model_args['type'] = BackboneName.ResNet18
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        spatial_path = SpatialPath(self.data_channel, 128,
                                   bn_name=self.bn_name, activation_name=self.activation_name)
        self.add_block_list(spatial_path.get_name(), spatial_path, 128)

        backbone = self.backbone_factory.get_backbone_model(self.model_args)
        base_out_channels = backbone.get_outchannel_list()
        self.add_block_list(BlockType.BaseNet, backbone, base_out_channels[-1])

        inter_channels = 128
        global_context = GlobalAvgPooling(512, inter_channels, bn_name=self.bn_name,
                                          activation_name=self.activation_name)
        self.add_block_list(global_context.get_name(), global_context, base_out_channels[-1])

        layer1 = RouteLayer('9')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer1.layers])
        self.add_block_list(layer1.get_name(), layer1, output_channel)

        context_path1 = ContextPath(512, inter_channels, bn_name=self.bn_name,
                                    activation_name=self.activation_name)
        self.add_block_list(context_path1.get_name(), context_path1, inter_channels)

        layer2 = RouteLayer('7')
        output_channel = sum([base_out_channels[i] if i >= 0
                              else self.block_out_channels[i] for i in layer2.layers])
        self.add_block_list(layer2.get_name(), layer2, output_channel)

        context_path2 = ContextPath(256, inter_channels, bn_name=self.bn_name,
                                    activation_name=self.activation_name)
        self.add_block_list(context_path2.get_name(), context_path2, inter_channels)

        ffm = FeatureFusionBlock(256, 256, 4, bn_name=self.bn_name,
                                 activation_name=self.activation_name)
        self.add_block_list(ffm.get_name(), ffm, 256)

        conv1 = ConvBNActivationBlock(in_channels=256,
                                      out_channels=64,
                                      kernel_size=3,
                                      padding=1,
                                      bias=False,
                                      bnName=self.bn_name,
                                      activationName=self.activation_name)
        self.add_block_list(conv1.get_name(), conv1, 64)

        dropout = nn.Dropout(0.1)
        self.add_block_list(LayerType.Dropout, dropout, self.block_out_channels[-1])

        conv2 = nn.Conv2d(64, self.class_number, 1)
        self.add_block_list(LayerType.Convolutional, conv2, self.class_number)

        layer = Upsample(scale_factor=8, mode='bilinear')
        self.add_block_list(layer.get_name(), layer, self.block_out_channels[-1])

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.CrossEntropy2dLoss,
                       'weight_type': 0,
                       'reduction': 'mean',
                       'ignore_index': 250}
        loss = self.loss_factory.get_loss(loss_config)
        self.add_block_list(loss.get_name(), loss, self.block_out_channels[-1])
        self.lossList.append(loss)

    def forward(self, x):
        base_outputs = []
        layer_outputs = []
        output = []
        for key, block in self._modules.items():
            if BiSeNetBlockName.SpatialPath in key:
                spatial_out = block(x)
                layer_outputs.append(spatial_out)
            else:
                if BlockType.BaseNet in key:
                    base_outputs = block(x)
                    x = base_outputs[-1]
                elif LayerType.RouteLayer in key:
                    x = block(layer_outputs, base_outputs)
                elif BiSeNetBlockName.ContextPath in key:
                    x = block(layer_outputs[-2], layer_outputs[-1])
                elif BiSeNetBlockName.FeatureFusionBlock in key:
                    x = block(layer_outputs[0], layer_outputs[-1])
                elif self.loss_factory.has_loss(key):
                    output.append(x)
                else:
                    x = block(x)
                layer_outputs.append(x)
            print(key, layer_outputs[-1].shape)
        return output

