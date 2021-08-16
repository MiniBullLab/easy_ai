#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.common.upsample_layer import Upsample
from easyai.model_block.utility.base_block import *


class PadimHead(BaseBlock):

    def __init__(self, layers):
        super().__init__(HeadType.PadimHead)
        self.layers = [int(x) for x in layers.split(',') if x.strip()]
        assert len(self.layers) > 0
        self.up = Upsample(scale_factor=2)

    def embedding_concat(self, x, y):
        # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def up_concat(self, x, y):
        z = self.up(y)
        return torch.cat([x, z], dim=1)

    def forward(self, layer_outputs, base_outputs):
        temp_layer_outputs = [layer_outputs[i] if i < 0 else base_outputs[i]
                              for i in self.layers]
        x = temp_layer_outputs[0]
        for feature in temp_layer_outputs[1:]:
            x = self.embedding_concat(x, feature)  # n, c, h, w
        return x


class PatchHead(BaseBlock):

    def __init__(self):
        super().__init__(HeadType.PatchHead)
        self.up = Upsample(scale_factor=2)

    def embedding_concat(self, x, y):
        # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def up_concat(self, x, y):
        z = self.up(y)
        return torch.cat([x, z], dim=1)

    def forward(self, embeddings):
        x = embeddings[0]
        y = embeddings[1]
        embedding = self.embedding_concat(x, y)  # n, c, h, w
        return embedding
