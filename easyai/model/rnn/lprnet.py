#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie
"""
LPRNet: License Plate Recognition via Deep Neural Networks
"""

from easyai.name_manager.model_name import ModelName
from easyai.name_manager.loss_name import LossName
from easyai.model.utility.base_classify_model import *
from easyai.model.utility.model_registry import REGISTERED_RNN_MODEL


class SmallBasicBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


@REGISTERED_RNN_MODEL.register_module(ModelName.LPRNet)
class LPRNet(BaseClassifyModel):
    def __init__(self, data_channel=3, class_number=68):
        super().__init__(data_channel, class_number)
        self.set_name(ModelName.LPRNet)
        self.backbone = None
        self.container = None
        self.create_block_list()

    def create_block_list(self):
        self.clear_list()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            SmallBasicBlock(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            SmallBasicBlock(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            SmallBasicBlock(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=256, out_channels=self.class_number, kernel_size=(13, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=self.class_number),
            nn.ReLU(),  # *** 22 ***
        )

        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_number,
                      out_channels=self.class_number,
                      kernel_size=(1, 1),
                      stride=(1, 1)),
            # nn.BatchNorm2d(num_features=self.class_number),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_number, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

        self.create_loss_list()

    def create_loss_list(self, input_dict=None):
        self.lossList = []
        loss_config = {'type': LossName.CTCLoss,
                       'blank_index': 0,
                       'reduction': 'mean',
                       'use_focal': False}
        loss = self.loss_factory.get_loss(loss_config)
        self.lossList.append(loss)

    def forward(self, x):
        output = []
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        x = logits.permute((0, 2, 1))
        output.append(x)
        return output
