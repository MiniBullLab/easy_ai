#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.model_block.utility.base_block import *


class PatchHead(BaseBlock):

    def __init__(self):
        super().__init__(HeadType.PatchHead)

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

    def forward(self, embeddings):
        x = embeddings[0]
        y = embeddings[1]
        embedding = self.embedding_concat(x, y)  # n, c, h, w
        return embedding
