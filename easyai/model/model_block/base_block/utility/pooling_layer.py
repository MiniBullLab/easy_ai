#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.base_name.block_name import LayerType, BlockType
from easyai.model.model_block.base_block.utility.base_block import *
from easyai.base_algorithm.roi_align import ROIAlign


class MyMaxPool2d(BaseBlock):

    def __init__(self, kernel_size, stride, ceil_mode=False):
        super().__init__(LayerType.MyMaxPool2d)
        layers = OrderedDict()
        maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                               padding=int((kernel_size - 1) // 2),
                               ceil_mode=ceil_mode)
        if kernel_size == 2 and stride == 1:
            layer1 = nn.ZeroPad2d((0, 1, 0, 1))
            layers["pad2d"] = layer1
            layers[LayerType.MyMaxPool2d] = maxpool
        else:
            layers[LayerType.MyMaxPool2d] = maxpool
        self.layer = nn.Sequential(layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class MyAvgPool2d(BaseBlock):

    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super().__init__(LayerType.MyAvgPool2d)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, ceil_mode)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class GlobalAvgPool2d(BaseBlock):
    def __init__(self):
        super().__init__(LayerType.GlobalAvgPool)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        return x
        # h, w = x.shape[2:]
        # if torch.is_tensor(h) or torch.is_tensor(w):
        #     h = np.asarray(h)
        #     w = np.asarray(w)
        #     return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))
        # else:
        #     return F.avg_pool2d(x, kernel_size=(h, w), stride=(h, w))


# SPP
class SpatialPyramidPooling(BaseBlock):
    def __init__(self, pool_sizes=(5, 9, 13)):
        super().__init__(BlockType.SpatialPyramidPooling)
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2)
                                       for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)
        return features


class MultiROIPooling(BaseBlock):

    def __init__(self, out_channels, output_size,
                 scales, sampling_ratio):
        """
        Arguments:
                output_size (list[tuple[int]] or list[int]): output size for the pooled region
                scales (list[float]): scales for each level
                sampling_ratio (int): sampling ratio for ROIAlign
        """
        super().__init__(BlockType.MultiROIPooling)
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.num_levels = len(poolers)
        self.poolers = nn.ModuleList(poolers)
        self.out_channels = out_channels
        self.output_size = output_size

        self.k_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        self.k_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.s0 = 224
        self.lvl0 = 4
        self.eps = 1e-6

    def compute_map_levels(self, box_list):
        TO_REMOVE = 1  # TODO remove
        boxes = box_list[:, :, :4]
        concat_boxes = boxes.view(-1, 4)
        width = concat_boxes[:, 2] - box_list[:, 0] + TO_REMOVE
        height = concat_boxes[:, 3] - box_list[:, 1] + TO_REMOVE
        area = torch.sqrt(width * height)
        target_lvls = torch.floor(self.lvl0 + torch.log2(area / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min

    def convert_to_roi_format(self, box_list):
        device, dtype = box_list.device, box_list.dtype
        boxes = box_list[:, :, :4]
        concat_boxes = boxes.view(-1, 4)
        ids = torch.cat([torch.full((len(b), 1), i, dtype=dtype, device=device)
                         for i, b in enumerate(box_list)],
                        dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, proposals):
        rois = self.convert_to_roi_format(proposals)
        if self.num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.compute_map_levels(proposals)
        num_rois = len(rois)

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, self.out_channels, self.output_size[0], self.output_size[1]),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)

        return result
